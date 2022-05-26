#! /usr/bin/env python3.8

if __name__ == '__main__':
    import sys
    import os

    #-- Check if a virtual environment was passed and activate via redirection.
    if len(sys.argv) == 4:
        if sys.argv[3] != '0':
            path = os.path.join(sys.argv[3], 'bin')
            #-- Put the venv in the OS PATH environment variable and run the this file with that python.
            if 'PATH' in os.environ:
                os.environ['PATH'] = path + os.pathsep + os.environ['PATH']
            else:
                os.environ['PATH'] = path
            #-- Execute file again
            os.execvp(__file__, sys.argv[:3])
            sys.exit()
        else:
            print('No venv passed.')
            sys.argv.pop(-1)

    from mpi4py import MPI
    import pickle
    import compress_pickle
    import socket
    import time
    import copy

    #-- Comment if running C++ branch
    from pybkb.common.bayesianKnowledgeBase import bayesianKnowledgeBase as BKB
    import pybkb.python_base.reasoning.reasoning as BKBR

    #-- Uncomment if running C++ Branch
    #import bayesianKnowledgeBase as BKB
    #import BKBReasoning as BKBR

    #
    # Notes regarding MPI issues:
    #   - Nonblocking isend/irecv limited to 32kb
    #   - Not safe to do an isend and change buffer contents especially of
    #       isend may not be completed. 

    #
    # This script implements BKBReasoning.computeJoint(...) for
    #   distributed computing
    #

    Debug = False

    MPI_Debug = False
    Socket_Debug = False
    SOCKET_PORT = None
    HEADERSIZE = BKBR.HEADERSIZE
    SOCKET_BUF_SIZE = BKBR.SOCKET_BUF_SIZE
    socket_full_msg = b""  # This packages own message buffer.


    def sendSocketMessage(msg_object, socket): 
        msg = compress_pickle.dumps(msg_object, compression="lz4")
        msg = bytes(f"{len(msg):<{HEADERSIZE}}", 'utf-8') + msg
        if Socket_Debug:
            if Debug:   
                print ('{} sending message {}'.format(WORKER_DESC, msg))
            else:
                print ('{} sending message {}'.format(WORKER_DESC, msg[:20]))
        socket.send(msg)

    def receiveSocketMessage(socket):
        global socket_full_msg
        while True: # Loop to get in the full message:
            if len(socket_full_msg) < HEADERSIZE:
                msg = socket.recv(SOCKET_BUF_SIZE)
                socket_full_msg += msg
                if Socket_Debug:
                    if Debug:
                        print ('{} receiving header information buffered {} + new {}'.format(WORKER_DESC, socket_full_msg, msg))
                    else:
                        print ('{} receiving header information'.format(WORKER_DESC))
            if len(socket_full_msg) < HEADERSIZE: # Need to get in the header first
                continue
            msglen = int(socket_full_msg[:HEADERSIZE])
            if Socket_Debug:
                print ('{} receiving message of length {}'.format(WORKER_DESC, msglen))
            socket_full_msg = socket_full_msg[HEADERSIZE:] # Prune from head
            while len(socket_full_msg) < msglen:
                msg = socket.recv(SOCKET_BUF_SIZE)
                if Socket_Debug:
                    if Debug:
                        print ('{} receiving message buffered {} + new {}'.format(WORKER_DESC, socket_full_msg, msg))
                socket_full_msg += msg
            msg = socket_full_msg[:msglen]
            socket_full_msg = socket_full_msg[msglen:]
            if Socket_Debug:
                if Debug:   
                    print ('{} received message {}'.format(WORKER_DESC, msg))
                    print ('{} remaining socket message {}'.format(WORKER_DESC, socket_full_msg))
                else:
                    print ('{} received message {}'.format(WORKER_DESC, msg[:20]))
                    print ('{} remaining socket message len {}'.format(WORKER_DESC, len(socket_full_msg)))
            return (compress_pickle.loads(msg, compression="lz4"))


    WORKER_ITEMS = 1000
        # Maximum number of items a worker processes froma single task item
        # i.e. the new items spawned by worker

    # The master sends items to each worker for processing
    # Each worker processes for a time and returns new items to be processed
    #   and completed inferences.

    comm = MPI.COMM_WORLD # My MPI communication channel
    comm_size = comm.Get_size() # Number of processes
    rank = comm.Get_rank() # My worker ID
    # rank == 0 master, all other ranks are workers
    WORKER_DESC = "Rank #{} on {}:".format(rank, socket.gethostname())

    # queue tuple of form:
    # ( probability, set of I-nodes supported, set of S-nodes used,
    #   set of incompatible snodes, set of unsupported I-nodes,
    #   set of unassigned components )
    global __idx_probability 
    global __idx_supported_inodes 
    global __idx_snodes_used 
    global __idx_assigned_components 
    global __idx_unsupported_inodes 
    global __idx_unassigned_components
    __idx_probability = 0
    __idx_supported_inodes = 1
    __idx_snodes_used = 2
    __idx_assigned_components = 3
    __idx_unsupported_inodes = 4
    __idx_unassigned_components = 5

    # MPI Message Tags
    MPITAG_worker_ready = 1000 
    MPITAG_worker_task = 5
    MPITAG_worker_answer = 99
    MPITAG_worker_answer_ready = 999
    MPITAG_worker_update = 777

    if len(sys.argv) < 3: # Improper usage
        sys.exit("{} Usage: {} <parent hostname> <socket port on parent>".format(WORKER_DESC, sys.argv[0]))

    if rank == 0: # Master
        # Results from master are as follows (or 'None' if errorful):
        #

        if MPI_Debug:
            print ('{} master process started.'.format(WORKER_DESC))
        start_time = time.time()

        # Set up socket communication to parent
        SOCKET_PORT = int(sys.argv[2])
        master_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if Socket_Debug:
            print ('{} attempting socket connection to {} on port {}....'.format(WORKER_DESC, sys.argv[1], SOCKET_PORT))
        master_socket.connect((sys.argv[1], SOCKET_PORT))
        if Socket_Debug:
            print ('{} connected.'.format(WORKER_DESC))


        if MPI_Debug:
            print ('{} connected to parent on {}.'.format(WORKER_DESC, sys.argv[1]))

        print ('{} number of workers = {}'.format(WORKER_DESC, comm_size))
        workers = [ w for w in range(1, comm_size) ] # Worker IDs
        _new_bkb = True
        _new_target_inodes = True
        _new_target_components = True
        PartialsCheck = False
        queue = list()
        worker_answers_ready = list()
        worker_ready = list()
        worker_ready_wid = list()
        reasoning_flag = False
        partials_explored = None
        completed_count = None
        total_explored = 0

        # Lets look for work!
        while True:
            # Lets work-loop!
            print ('\r                                                                          ', end='')
            print ('\r# of all: {} \t partials: {} \t completed: {}'.format(total_explored, partials_explored, completed_count), end='')
            while len(queue) > 0 or len(worker_answers_ready) > 0:
                if len(queue) > 0 and len(worker_ready) > 0: # Is a worker ready for a reasoning tasks?
                    if Debug:
                        print ('{} checking to see if worker available for reasoning task.'.format(WORKER_DESC)) 
                    rid, flag, wid = MPI.Request.testany(worker_ready)
                        # Is there a worker ready?
                    if flag: # A worker is ready
                        if MPI_Debug:
                            print ('{} worker flag {} Rank #{} from position {} in worker queue {}'.format(WORKER_DESC, flag, wid, rid, worker_ready))
                        worker_ready.pop(rid) # Remove from list
                        _wid = worker_ready_wid.pop(rid)
                        if wid != _wid:
                            sys.exit('{} wid {} does not match retrieved ready wid {}!'.format(WORKER_DESC, wid, _wid))
                        item = queue.pop(0) # Pop item to send to workers
                        if MPI_Debug:
                            if Debug:
                                print ('{} dispatching to worker Rank #{} task {}.'.format(WORKER_DESC, wid, item))
                            else:
                                print ('{} dispatching task to worker Rank #{}.'.format(WORKER_DESC, wid))
                        _item = compress_pickle.dumps(item, compression="lz4")
                        comm.send(_item, dest = wid, tag = MPITAG_worker_task)
                        if MPI_Debug:
                            print ('{} requeueing worker ANSWER ready for Rank #{}'.format(WORKER_DESC, wid))
                        worker_answers_ready.append(comm.irecv(source = wid, tag = MPITAG_worker_answer_ready))
                        continue # Prioritize dispatch over receiving answers
                    
                # Check to see if any answers have been returned by workers
                if len(worker_answers_ready) == 0:
                    continue
    #            if MPI_Debug:
    #                print ('{} worker_answers_ready request queued # = {}'.format(WORKER_DESC, len(worker_answers_ready)))
                rid, flag, wid = MPI.Request.testany(worker_answers_ready)
                    # worker_answer = ( processed_list, completed_inferences, queue ) 
                if flag:  # A worker has an answer
                    if MPI_Debug:
                        print ('{} worker ANSWER flag {} Rank #{} from position {} in worker queue {}'.format(WORKER_DESC, flag, wid, rid, worker_answers_ready))
                    worker_answers_ready.pop(rid)
                    __worker_answer = comm.recv(source = wid, tag = MPITAG_worker_answer)
                    worker_answer = compress_pickle.loads(__worker_answer, compression="lz4")
                        # Using recv instead of irecv due to buffer limitations of MPI
                    if MPI_Debug:
                        if Debug:
                            print ('{} Answer from worker Rank #{} is {}'.format(WORKER_DESC, wid, worker_answer))
                        else:
                            print ('{} Received answer from worker Rank #{}'.format(WORKER_DESC, wid))
                        if PartialsCheck:
                            print ('{} # of returned partials explored = {}'.format(WORKER_DESC, len(worker_answer[0])))
                        else:
                            print ('{} # of returned partials explored = {}'.format(WORKER_DESC, worker_answer[0]))
                        print ('{} # of returned complete inferences = {}'.format(WORKER_DESC, len(worker_answer[1])))
                        print ('{} # of returned unprocessed partials = {}'.format(WORKER_DESC, len(worker_answer[2])))
                        print ('{} requeueing worker ready for Rank #{}'.format(WORKER_DESC, wid))
                    worker_ready.append(comm.irecv(source = wid, tag = MPITAG_worker_ready))    
                    worker_ready_wid.append(wid) # Parallels worker_ready queue


                    # Process partials
                    if PartialsCheck:
                        for item in worker_answer[0]:
                            partials_explored += 1
                            if Debug:
                                print ('{} ----------------------- Processed Partial #{}'.format(WORKER_DESC, partials_explored))
                                BKBR.printInference(bkb, item, WORKER_DESC)
                            partial = ( frozenset(item[__idx_supported_inodes]), frozenset(item[__idx_unsupported_inodes]) )
                            if len(item[__idx_unsupported_inodes]) > 0 and partial in partials:
                                print ('{} ERROR -- Partial has already been explored!'.format(WORKER_DESC))
                                print ('{} Supported --'.format(WORKER_DESC))
                                BKBR.printINodeTuples (bkb, partial[0], prefix = WORKER_DESC)
                                print ('{} Unsupported --'.format(WORKER_DESC))
                                BKBR.printINodeTuples (bkb, partial[1], prefix = WORKER_DESC)
                            else:
                                partials.add(partial)
                    else:
                        if worker_answer[0] != None:
                            partials_explored += int(worker_answer[0])
                    total_explored += partials_explored
        
        
                    # Process completed inferences
                    if worker_answer[1] != None:
                        completed_count += len(worker_answer[1])
                        total_explored += completed_count
                    for item in worker_answer[1]:
                        if Debug:
                            print ('{} =========Inference built.'.format(WORKER_DESC))
                            BKBR.printInference(bkb, item, WORKER_DESC)
                        answer = frozenset(item[__idx_supported_inodes] & target_inodes_filter)
                        try:
                            probabilities[answer] += item[__idx_probability]
                        except KeyError:
                            probabilities[answer] = item[__idx_probability]
                            completed_inferences[answer] = list()
                            contributions[answer] = {}
                        item[__idx_snodes_used] = list(BKBR.flatten_list(item[__idx_snodes_used]))
                        completed_inferences[answer].append(item)
                        for s_idx in item[__idx_snodes_used]:
                            try:
                                contributions[answer][s_idx] += item[__idx_probability]
                            except KeyError:
                                contributions[answer][s_idx] = item[__idx_probability]
            
                    # Enqueue unprocessed partials
                    for up_idx, item in enumerate(worker_answer[2]):
                        if Debug:
                            print ('{} ========= Unprocessed Partial #{}'.format(WORKER_DESC, up_idx))
                            BKBR.printInference(bkb, item, WORKER_DESC)
                        queue.insert(0, item)


            if reasoning_flag: # Just finished reasoning
                # Wait for them to finish to collect answers

                answer = [ probabilities, contributions, completed_inferences, partials_explored, time.time() - start_time ]
                if Debug:
                    print ('{} Final Answer - {}'.format(WORKER_DESC, answer))
                if Socket_Debug:
                    print ('{} send final answer to parent.'.format(WORKER_DESC))
                sendSocketMessage(answer, master_socket)

                # Otherwise -- 
                # Clear workers caches

                if MPI_Debug:
                    print ('{} clearing worker caches.'.format(WORKER_DESC))
                    print ('{} ready worker remaining queue = {}'.format(WORKER_DESC, len(worker_ready)))
                _worker_ready_wid = set(worker_ready_wid)
                for wid in workers:
                    if wid in _worker_ready_wid:
                        continue # Already queued up
                    if MPI_Debug:
                        print ('{} requeueing worker ready for Rank #{}'.format(WORKER_DESC, wid))
                    worker_ready.append(comm.irecv(source = wid, tag = MPITAG_worker_ready))
                    worker_ready_wid.append(wid)

                while len(worker_ready) > 0:
                    rid, flag, wid = MPI.Request.testany(worker_ready)
                    if not flag:
                        continue
                    if MPI_Debug:
                        print ('{} worker flag {} Rank #{} from position {} in worker queue {}'.format(WORKER_DESC, flag, wid, rid, worker_ready))
                    worker_ready.pop(rid)
                    _wid = worker_ready_wid.pop(rid)
                    if wid != _wid:
                        sys.exit('{} wid {} does not match retrieved ready wid {}!'.format(WORKER_DESC, wid, _wid))
                    comm.send('Clear', dest = wid, tag = MPITAG_worker_task)
                        # Clear the worker's cache
                if Debug:
                    print ('{} worker caches all cleared.'.format(WORKER_DESC))
                reasoning_flag = False


            # Wait for the next command
            if Socket_Debug:
                print ('{} waiting for command from parent.'.format(WORKER_DESC))
            cmd = receiveSocketMessage(master_socket)
            if Socket_Debug:
                print ('{} received message {} from parent.'.format(WORKER_DESC, cmd))
            if cmd == 'Exit': # Shutdown
                break
        
            if cmd == 'New BKB': # get BKB and update workers
                _new_bkb = True
                bkb = receiveSocketMessage(master_socket)
                if Socket_Debug:
                    print ('{} received BKB from parent.'.format(WORKER_DESC))
                target_inodes = list()
                target_components = list()
                continue

            if cmd == 'New Target I-Nodes': # Set up new I-Nodes as targets
                _new_target_inodes = True
                target_inodes = receiveSocketMessage(master_socket)
                if Socket_Debug:
                    print ('{} received target i-nodes from parent.'.format(WORKER_DESC))
                continue

            if cmd == 'New Target Components': # Set up new components as targets
                _new_target_components = True
                target_components = receiveSocketMessage(master_socket)
                if Socket_Debug:
                    print ('{} received target components from parent.'.format(WORKER_DESC))
                continue

            if cmd == 'Set PartialsCheck': # Set PartialsCheck to True
                PartialsCheck = True
                continue

            if cmd == 'Unset PartialsCheck': # Set PartialCheck to False
                PartialsCheck = False
                continue

            if cmd == 'Start': # Start problem.
                abort_reasoning = False
                    # This flag can be set to True in case the problem setup is
                    #   is inconsistent. If True, setup continues but no 
                    #   reasoning items are queued and hence an empty reasoning 
                    #   result.

                if _new_bkb:
                    _new_bkb = False
                    BKBR.setup(bkb)

                    # Otherwise -- 
                    #   Update worker BKBs
                    if MPI_Debug:
                        print ('{} updating worker BKBs.'.format(WORKER_DESC))
                        print ('{} ready worker remaining queue = {}'.format(WORKER_DESC, len(worker_ready)))
                    _worker_ready_wid = set(worker_ready_wid)
                    for wid in workers:
                        if wid in _worker_ready_wid:
                            continue # Already queued up
                        if MPI_Debug:
                            print ('{} requeueing worker ready for Rank #{}'.format(WORKER_DESC, wid))
                        worker_ready.append(comm.irecv(source = wid, tag = MPITAG_worker_ready))
                        worker_ready_wid.append(wid)

                    __bkb = compress_pickle.dumps(bkb, compression="lz4")
                    while len(worker_ready) > 0:
                        rid, flag, wid = MPI.Request.testany(worker_ready)
                        if not flag:
                            continue
                        if MPI_Debug:
                            print ('{} worker flag {} Rank #{} from position {} in worker queue {}'.format(WORKER_DESC, flag, wid, rid, worker_ready))
                        worker_ready.pop(rid)
                        _wid = worker_ready_wid.pop(rid)
                        if wid != _wid:
                            sys.exit('{} wid {} does not match retrieved ready wid {}!'.format(WORKER_DESC, wid, _wid))
                        comm.send('Update BKB', dest = wid, tag = MPITAG_worker_task)
                        comm.send(__bkb, dest = wid, tag = MPITAG_worker_update)
                    if MPI_Debug:
                        print ('{} updated worker BKBs.'.format(WORKER_DESC))

                if MPI_Debug:
                    print ('{} updating workers\' PartialsCheck flag'.format(WORKER_DESC))
                _worker_ready_wid = set(worker_ready_wid)
                for wid in workers:
                    if wid in _worker_ready_wid:
                        continue # Already queued up
                    if MPI_Debug:
                        print ('{} requeueing worker ready for Rank #{}'.format(WORKER_DESC, wid))
                    worker_ready.append(comm.irecv(source = wid, tag = MPITAG_worker_ready))
                    worker_ready_wid.append(wid)

                while len(worker_ready) > 0:
                    rid, flag, wid = MPI.Request.testany(worker_ready)
                    if not flag:
                        continue
                    if MPI_Debug:
                        print ('{} worker flag {} Rank #{} from position {} in worker queue {}'.format(WORKER_DESC, flag, wid, rid, worker_ready))
                    worker_ready.pop(rid)
                    _wid = worker_ready_wid.pop(rid)
                    if wid != _wid:
                        sys.exit('{} wid {} does not match retrieved ready wid {}!'.format(WORKER_DESC, wid, _wid))
                    comm.send('Update PartialsCheck', dest = wid, tag = MPITAG_worker_task)
                    comm.send(PartialsCheck, dest = wid, tag = MPITAG_worker_update)
                if MPI_Debug:
                    print ('{} workers\' PartialsCheck updated.'.format(WORKER_DESC))

                if _new_target_inodes:
                    _new_target_inodes = False
                    # Check to make sure there are no imcompatible target_inodes
                    common = set()
                    assigned_components = set()
                    for inode_tuple in target_inodes:
                        c_idx = inode_tuple[0]
                        assigned_components.add(c_idx)
                        common = bkb.__BKBR_inodes_in_component[c_idx] & target_inodes
                        if len(common) > 1: # Another I-node found
                            print ('{} Conflicting I-nodes found in target_inodes.'.format(WORKER_DESC))
                            while len(common) > 0:
                                i_tuple = common.pop()
                                print ('{}\t < {} > ='.format(WORKER_DESC, bkb.getComponentName(i_tuple[0])), end='')
                                print ('\t< {} > = '.format(bkb.getComponentINodeName(i_tuple[0], i_tuple[1])))
                            abort_reasoning = True

                if _new_target_components:
                    _new_target_components = False
                    # Fix target_components if assigned in target_inodes
                    # also, make target_inodes
                    target_inodes_filter = copy.copy(target_inodes)
                    fixed_target_components = set()
                    for c_idx in target_components:
                        common = bkb.__BKBR_inodes_in_component[c_idx] & target_inodes
                        if len(common) > 0:
                            continue # Skip. Already assigned
                        fixed_target_components.add(c_idx)
                        target_inodes_filter |= bkb.__BKBR_inodes_in_component[c_idx]

                # Begin search for inferences 
                contributions = {}
                probabilities = {}
                completed_inferences = {}
                partials_explored = 0
                completed_count = 0
                if PartialsCheck:
                    partials = set()
            
                # queue tuple of form:
                # ( probability, set of I-nodes supported, set of S-nodes used,
                #   set of incompatible snodes, set of unsupported I-nodes,
                #   set of unassigned components )
                #
                #   set of I-nodes supported implies it is supported by some S-nodes used
                if not abort_reasoning: # Abort queueing
                    start = [ 1.0, set(), list(), copy.copy(assigned_components), copy.copy(target_inodes), copy.copy(fixed_target_components) ]
            
                    queue = list()
                    queue.append(start)

           
                if len(worker_answers_ready) > 0:
                    sys.exit('{} worker tasks not complete!'.format(WORKER_DESC)) 
                if MPI_Debug:
                    print ('{} ready worker remaining queue = {}'.format(WORKER_DESC, len(worker_ready)))
                _worker_ready_wid = set(worker_ready_wid)
                for wid in workers:
                    if wid in _worker_ready_wid:
                        continue # Already waiting
                    if MPI_Debug:
                        print ('{} requeueing worker ready for Rank #{}'.format(WORKER_DESC, wid))
                    worker_ready.append(comm.irecv(source = wid, tag = MPITAG_worker_ready))
                    worker_ready_wid.append(wid)

                reasoning_flag = True
                continue
           
            if not cmd == None:
                print ('{} received unknown command {}.'.format(WORKER_DESC, cmd))
                continue


        # All done
       
        if MPI_Debug:
            print ('{} stopping the workers.'.format(WORKER_DESC))
            print ('{} ready worker remaining queue = {}'.format(WORKER_DESC, len(worker_ready)))
        _worker_ready_wid = set(worker_ready_wid)
        for wid in workers:
            if wid in _worker_ready_wid:
                continue # Already queued up
            if MPI_Debug:
                print ('{} requeueing worker ready for Rank #{}'.format(WORKER_DESC, wid))
            worker_ready.append(comm.irecv(source = wid, tag = MPITAG_worker_ready))
            worker_ready_wid.append(wid)

        while len(worker_ready) > 0:
            rid, flag, wid = MPI.Request.testany(worker_ready)
            if not flag:
                continue
            if MPI_Debug:
                print ('{} worker flag {} Rank #{} from position {} in worker queue {}'.format(WORKER_DESC, flag, wid, rid, worker_ready))
            worker_ready.pop(rid)
            _wid = worker_ready_wid.pop(rid)
            if wid != _wid:
                sys.exit('{} wid {} does not match retrieved ready wid {}!'.format(WORKER_DESC, wid, _wid))
            comm.send(None, dest = wid, tag = MPITAG_worker_task)

        # Close the sockets and exit.
        master_socket.close()
        if MPI_Debug:
            print('{} normal exit.'.format(WORKER_DESC))
        sys.exit(0)

    #======================================================================

    else: # Worker

        PartialsCheck = False

        if MPI_Debug:
            print ('{} worker process started.'.format(WORKER_DESC))
        start_time = time.time()

        while True: # Worker is ready
            if MPI_Debug:
                print ('{} worker ready and waiting.'.format(WORKER_DESC))
            comm.send(rank, dest = 0, tag = MPITAG_worker_ready)
            if MPI_Debug:
                print ('{} worker waiting for tasks.'.format(WORKER_DESC))
            item = comm.recv(source = 0, tag = MPITAG_worker_task)
            if MPI_Debug:
                if Debug:
                    print ('{} received work item {}'.format(WORKER_DESC, item))
                else:
                    print ('{} received work item'.format(WORKER_DESC))

            if item == None: # No more tasks. Worker terminates
                if Debug:
                    print('{} terminated by master.'.format(WORKER_DESC))
                sys.exit(0)

            if item == 'Clear': # Reset local caches
                if Debug:
                    print ('{} clear caches.'.format(WORKER_DESC))
                if PartialsCheck:   
                    processed = list()
                __BKBR_snode_tail = {}
                __BKBR_snodes_by_head = {} 
                __BKBR_snodes_by_tail = {} 
                continue

            if item == 'Update BKB': # Load in new BKB
                if Debug:
                    print ('{} updating BKB.'.format(WORKER_DESC))
                __bkb = comm.recv(source = 0, tag = MPITAG_worker_update)
                bkb = compress_pickle.loads(__bkb, compression="lz4")
                partials = set()
                __BKBR_snode_tail = {}
                __BKBR_snodes_by_head = {} 
                __BKBR_snodes_by_tail = {} 
                if Debug:
                    print ('{} BKB updated.'.format(WORKER_DESC))
                continue

            if item == 'Update PartialsCheck':
                if Debug:
                    print ('{} updating PartialsCheck'.format(WORKER_DESC))
                PartialsCheck = comm.recv(source = 0, tag = MPITAG_worker_update)
                if Debug:
                    print ('{} PartialsCheck updated'.format(WORKER_DESC))
                continue

            # Must be a working task item
            item = compress_pickle.loads(item, compression="lz4")

            # queue tuple of form:
            # ( probability, set of I-nodes supported, set of S-nodes used,
            #   set of incompatible snodes, set of unsupported I-nodes,
            #   set of unassigned components )

            queue = list()
            queue.append(item)
            completed_inferences = list()
            partials_explored = 0
            if PartialsCheck:   
                processed = list()

            while (len(queue) > 0 and partials_explored < WORKER_ITEMS) or (len(queue) == 1 and partials_explored >= WORKER_ITEMS):
                item = queue.pop(0)
                partials_explored += 1

                if Debug:
                    print ('{} ----------------------- Partial #{}'.format(WORKER_DESC, partials_explored))
                    BKBR.printInference(bkb, item, WORKER_DESC)
                if PartialsCheck:
                    processed.append(item)

                if len(item[__idx_unsupported_inodes]) == 0 and len(item[__idx_unassigned_components]) == 0: # Finished
                    completed_inferences.append(item)

                else: # Continue expanding and do branches
                    new_open_inodes = copy.copy(item[__idx_unsupported_inodes])
                    # select an open I-node if any and requeue possible s-nodes
                    if len(new_open_inodes) > 0:
                        inode_tuple = new_open_inodes.pop()
                        if Debug:
                            print ('{} ======Processing I-node '.format(WORKER_DESC), end='')
                            BKBR.printINodeTuple(bkb, inode_tuple)
        
                        try:
                            candidate_snodes = __BKBR_snodes_by_head[inode_tuple]
                        except KeyError: # Make the local reduced version
                            __BKBR_snodes_by_head[inode_tuple] = bkb.__BKBR_snodes_by_head[inode_tuple]
                            candidate_snodes = __BKBR_snodes_by_head[inode_tuple]
                        if Debug:
                            print ('{} =====candidate snodes '.format(WORKER_DESC), end='')
                            print (candidate_snodes)
                        for s_idx in candidate_snodes: # Build branches
    #                        if Debug:
    #                            bkb.__BKBR_snodes[s_idx].output(bkb)
        
                            new_open_components = copy.copy(item[__idx_unassigned_components])
                            _new_open_inodes = copy.copy(new_open_inodes)
                            new_assigned_components = copy.copy(item[__idx_assigned_components])
        
                            # Check S-node compatibility
                            snode = bkb.__BKBR_snodes[s_idx]
                            try:
                                __BKBR_snode_tail[s_idx]
                            except KeyError:
                                __BKBR_snode_tail[s_idx] = snode.tail
        
                            fail = False
                            for (new_cidx, new_iidx) in __BKBR_snode_tail[s_idx]:
                                if ( new_cidx, new_iidx ) in _new_open_inodes or ( new_cidx, new_iidx ) in item[__idx_supported_inodes]:
                                    continue
                                if new_cidx in new_assigned_components:
                                    fail = True
                                    break
                                _new_open_inodes.add(( new_cidx, new_iidx))
                                new_assigned_components.add(new_cidx)
                                try:
                                    new_open_components.remove(new_cidx)
                                except KeyError:
                                    pass
                            if fail: # Incompatible
                                del _new_open_inodes
                                del new_assigned_components
                                del new_open_components
                                continue
        
                            new_prob = item[__idx_probability] * snode.probability
                            new_inodes_supported = copy.copy(item[__idx_supported_inodes])
                            new_inodes_supported.add(inode_tuple)
                            new_snodes_used = [ s_idx ]
                            new_snodes_used.append(item[__idx_snodes_used])
        
        
                            new_item = [ new_prob, new_inodes_supported, new_snodes_used, new_assigned_components, _new_open_inodes, new_open_components ]
                            if Debug:
                                print ('{} +++++++++++Spawning for S-node index = {}'.format(WORKER_DESC, s_idx))
                            queue.insert(0, new_item)
        
                    # select an open component and requeue posibble I-nodes
                    else:
                        new_open_components = item[__idx_unassigned_components]
                        found = False
                        while len(new_open_components) > 0:
                            c_idx = new_open_components.pop()
                            if not c_idx in item[__idx_assigned_components]:
                                found = True
                                break
                        if found:
                            for ( _, inode_i_idx ) in bkb.__BKBR_inodes_in_component[c_idx]:
                                inode_tuple = ( c_idx, inode_i_idx )
                                new_inodes = set()
                                new_inodes.add(inode_tuple)
                                new_assigned_components = item[__idx_assigned_components]
                                new_assigned_components.add(c_idx)
                            if Debug:
                                print ('{} +++++++++++Spawning for I-node = '.format(WORKER_DESC), end='')
                                printINodeTuple(bkb, inode_tuple)
                            new_item = [ item[__idx_probability], item[__idx_supported_inodes], item[__idx_snodes_used], new_assigned_components, new_inodes, new_open_components ]
                            queue.insert(0, new_item)
                        else:
                            new_item = [ item[__idx_probability], item[__idx_supported_inodes], item[__idx_snodes_used], new_assigned_components, set(), new_open_components ]
                            queue.insert(0, new_item)
                if Debug:
                    print ('{} worker finished reasoning.'.format(WORKER_DESC))

            # Worker finished
            if Debug:
                print ('{} !!! Reasoning task finished.'.format(WORKER_DESC))
            if MPI_Debug:
                print ('{} Answer ready'.format(WORKER_DESC))
            comm.send(rank, dest = 0, tag = MPITAG_worker_answer_ready)
            if PartialsCheck:
                answer = ( processed, completed_inferences, queue )
            else:
                answer = ( partials_explored, completed_inferences, queue ) 
            __answer = compress_pickle.dumps(answer, compression="lz4")
            if MPI_Debug:
                if Debug:
                    print ('{} sending answer {}'.format(WORKER_DESC, answer))
                else:
                    print ('{} sending answer'.format(WORKER_DESC))
            comm.send(__answer, dest = 0, tag = MPITAG_worker_answer)
            if MPI_Debug:
                print ('{} answer sent'.format(WORKER_DESC))
