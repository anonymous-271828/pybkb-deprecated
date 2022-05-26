import sys
import copy
import collections
import os
import pickle
import compress_pickle
import socket
import subprocess
import netifaces
import uuid
import time
import tqdm
import types
import logging
import pprint
pp = pprint.PrettyPrinter(indent=2)
#from progress.spinner import Spinner
from collections import defaultdict

#-- Setup logging
logger_std = logging.getLogger(__name__)
logger_socket = logging.getLogger('{}.socket'.format(__name__))
logger_std.setLevel(logging.WARNING)
logger_socket.setLevel(logging.WARNING)

#Uncomment to run in C++ branch
#import bayesianKnowledgeBase as BKB
#import BKBReasoning_MBLP as BKBR_MBLP
#import BKBR_MPIinference

#-- Comment if running in C++ branch
from pybkb.common.bayesianKnowledgeBase import bayesianKnowledgeBase as BKB
import pybkb.python_base.reasoning.BKBR_MPIinference as BKBR_MPIinference
#-- Commenting out until prepared to integrate with docplex
#import pybkb.python_base.BKBReasoning_MBLP as BKBR_MBLP
from .reasoning_result import UpdatingResult, RevisionResult

Debug = False
MPI_EXEC = '/usr/bin/mpiexec'
# Set up socket for distributed computing if needed
Socket_Debug = False
DISTRIBUTED_SETUP = False
SOCKET_PORT = None
HEADERSIZE = 10
SOCKET_BUF_SIZE = 1024
BKBR_socket = None
BKBR_clientsocket = None
BKBR_process = None
socket_full_msg = b"" # Message buffer

def setupDistributedReasoning(num_processes_per_host, hosts_file_name, venv=0):
    global BKBR_socket
    global BKBR_clientsocket
    global BKBR_process
    global DISTRIBUTED_SETUP
    global SOCKET_PORT
    if DISTRIBUTED_SETUP:
        return
    if logger_socket.level == 10:
        logger_socket.debug('Parent: Starting BKBR_MPIinference...')

    netifaces.ifaddresses('eno1')
    ip = netifaces.ifaddresses('eno1')[netifaces.AF_INET][0]['addr']
    BKBR_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    BKBR_socket.bind((ip, 0))
    ( _, SOCKET_PORT ) = BKBR_socket.getsockname()

    if logger_socket.level == 10:
        logger_socket.debug('Parent: Established socket at ( {} , {} )'.format(ip, SOCKET_PORT))
    BKBR_socket.listen()
    executable = os.path.abspath(BKBR_MPIinference.__file__)
    args = [ MPI_EXEC, "-N", str(num_processes_per_host), "--hostfile", hosts_file_name, executable, ip, str(SOCKET_PORT), str(venv) ]

    if logger_socket.level == 10:
        logger_socket.debug('')
        logger_socket.debug('')
        logger_socket.debug('')
        logger_socket.debug(executable)
        logger_socket.debug('{} -N {} --hostfile {} {} {} {}'.format(MPI_EXEC, num_processes_per_host, hosts_file_name, executable, ip, SOCKET_PORT))

    BKBR_process = subprocess.Popen(args)
    BKBR_clientsocket, address = BKBR_socket.accept()

    if logger_socket.level == 10:
        logger_socket.debug('Connection from {} has been established.'.format(address))
    DISTRIBUTED_SETUP = True
    return


def shutdownDistributedReasoning():
    global BKBR_socket
    global BKBR_process
    global DISTRIBUTED_SETUP
    if not DISTRIBUTED_SETUP:
        return
    if logger_socket.level == 10:
        logger_socket.debug('Shutting down BKBR_MPIinference...')
    BKBR_process.kill()
    BKBR_socket.close()
    DISTRIBUTED_SETUP = False


def sendSocketMessage(msg_object):
    global BKBR_clientsocket
    msg = compress_pickle.dumps(msg_object, compression="lz4")
    msg = bytes(f"{len(msg):<{HEADERSIZE}}", 'utf-8') + msg
    if logger_std.level == 10:
        logger_std.debug('Parent: sending message {}'.format(msg))
    if logger_socket.level == 10:
        logger_socket.debug('Parent: sending message {}.'.format(msg[0:20]))
    BKBR_clientsocket.send(msg)

def receiveSocketMessage():
    global socket_full_msg
    global BKBR_clientsocket
    while True: # Loop to get in the full message:
        if len(socket_full_msg) < HEADERSIZE:
            msg = BKBR_clientsocket.recv(SOCKET_BUF_SIZE)
            socket_full_msg += msg
        if logger_std.level == 10:
            logger_std.debug('Parent: receiving header information buffered {} + new {}'.format(socket_full_msg, msg))
        if logger_socket.level == 10:
            logger_socket.debug('Parent: receiving header information.')
        if len(socket_full_msg) < HEADERSIZE: # Need to get in the header first
            continue
        msglen = int(socket_full_msg[:HEADERSIZE])

        if logger_socket.level == 10:
            logger_socket.debug('Parent: receiving message of length {}'.format(msglen))
        socket_full_msg = socket_full_msg[HEADERSIZE:] # Prune from head
        while len(socket_full_msg) < msglen:
            msg = BKBR_clientsocket.recv(SOCKET_BUF_SIZE)
            if logger_std.level == 10:
                logger_std.debug('Parent: receiving message buffered {} + new {}'.format(socket_full_msg, msg))
            socket_full_msg += msg
        msg = socket_full_msg[:msglen]
        socket_full_msg = socket_full_msg[msglen:]
        if logger_std.level == 10:
            logger_std.debug('Parent: received message {}'.format(msg))
            logger_std.debug('Parent: remaining socket message {}'.format(socket_full_msg))
        if logger_socket.level == 10:
            logger_socket.debug('Parent: received message {}'.format(msg[:20]))
            logger_socket.debug('Parent: remaining socket message len {}'.format(len(socket_full_msg)))

        return (compress_pickle.loads(msg, compression="lz4"))
    

# Tuples used:
#    I-node ( component index, state index )
#    S-node ( S-node index )

# Note - a Causal Rule Set (CRS) for component X is simply all the S-nodes that
#   have as head, some I-node for X. Not about complementary rules set
#   Though we do need to compute those for constraints.
def process_evidence(evidence_dict, bkb):
    evidence = list()
    for comp_name, state_name in evidence_dict.items():
        comp_idx = bkb.getComponentIndex(comp_name)
        state_idx = bkb.getComponentINodeIndex(comp_idx, state_name)
        evidence.append((comp_idx, state_idx))
    return evidence

def process_targets(target_list, bkb):
    targets = list()
    for target_name in target_list:
        comp_idx = bkb.getComponentIndex(target_name)
        targets.append(comp_idx)
    return targets

'''
A wrapper function to call updating in the same format as the cpp reasoner is set up to do.
'''
def updating(bkb, evidence, targets, PartialsCheck=False, num_processes_per_host = 0, hosts_filename = None, updateBKB = True, venv=0, timeout=None):

    def process_results(res):
        """Transform generator lists in normal picklable lists.

        Only needed currently for complete_inferences portion of result.

        Parameters
        ----------

        res: tuple, required
            Result dictionary from update with the following structure:
                (probabilities, contributions, completed_inferences, partials_explored, compute_time)

        Returns
        -------

        res: tuple
            Returns process_updateslt but with all generator objects transformed to a normal list.
        """
        completed_inferences = defaultdict(list)
        #try:
        for target_inode, supported_inodes_inferences_dict in res[2].items():
            if len(supported_inodes_inferences_dict) == 0:
                completed_inferences[target_inode] = None
                continue
            for supported_inodes, inference_objs in supported_inodes_inferences_dict.items():
                for inference_obj in inference_objs:
                    inference_dict = {
                            "prob": inference_obj[0],
                            "supported_inodes": inference_obj[1],
                            "assigned_comps": inference_obj[3],
                            "unsupported_inodes": inference_obj[4],
                            "unassigned_components": inference_obj[5]
                            }
                    # Extract S-nodes
                    snodes = [bkb.__BKBR_snodes[snode_idx] for snode_idx in inference_obj[2]]
                    inference_dict["snodes_used"] = snodes

                    # Append to completed inferences
                    completed_inferences[target_inode].append(inference_dict)
        #except:
        #    logger_std.info('Complete inferences was not in expected format. So skipping generator removal.')
        #    completed_inferences = None    
        # Cast to list
        res = list(res)
        # Overwrite completed inferences in res
        res[2] = completed_inferences
        return tuple(res)


    res = updateComponents(bkb,
                           process_evidence(evidence, bkb),
                           process_targets(targets, bkb),
                           PartialsCheck=PartialsCheck,
                           num_processes_per_host=num_processes_per_host,
                           hosts_filename=hosts_filename,
                           updateBKB=updateBKB,
                           venv=venv,
                           timeout=timeout)
    
    #pp.pprint(res[2][(35,0)])
    #print(extractCompletedInferences([res])[1])
    res = process_results(res)
    #print(res[2])
    #-- Comment to run in BKB C++ branch
    result = UpdatingResult(res, bkb, evidence=evidence, targets=targets, snode_list=bkb.__BKBR_snodes)
    
    return result
    #-- Uncomment to run in BKB C++ branch
    #return res

def revision(bkb, evidence, PartialsCheck=False, num_processes_per_host = 0, hosts_filename = None, updateBKB = True, venv=0):
    res = revise(bkb,
           process_evidence(evidence, bkb),
           PartialsCheck=PartialsCheck,
           num_processes_per_host=num_processes_per_host,
           hosts_filename=hosts_filename,
           updateBKB=updateBKB,
           venv=venv)
    #res = process_results(res)
    #print(res[2])
    #-- Comment to run in BKB C++ branch
    result = RevisionResult(res, bkb, evidence=evidence, snode_list=bkb.__BKBR_snodes)
    
    return result
    #-- Uncomment to run in BKB C++ branch
    #return res


# queue tuple of form:
# ( probability, set of I-nodes supported, set of S-nodes used,
#   set of incompatible snodes, set of unsupported I-nodes,
#   set of unassigned components )
__idx_probability = 0
__idx_supported_inodes = 1
__idx_snodes_used = 2
__idx_assigned_components = 3
__idx_unsupported_inodes = 4
__idx_unassigned_components = 5


def selectComponent(bkb): # Returns a component index
    ans = input('Enter filter or * for all (wildcards not enabled) or \'enter\' to abort:')
    if ans == '':
        return (None)
    if ans == '*':
        indices = bkb.getAllComponentIndices()
    else:
        indices = list(bkb.getComponentsContains(ans))
    if len(indices) == 0:
        return (None)
    indices.sort()
    print ('Components --')
    for c_idx in indices:
        comp_name = bkb.getComponentName(c_idx)
        print ('\t' + str(c_idx + 1) + '\t---\t' + comp_name)
    try:
        choice = int(input('Select component index (or -1 to abort):'))
    except ValueError:
        choice = -1
    if (choice == -1) or (not (choice - 1) in indices):
        choice = None
    del indices
    return (choice)


def selectComponents(bkb): # Returns a set of component indices
    indices = set()
    while True:
        print (len(indices), end=' ')
        print ('components have been selected.')
        printComponents(bkb, indices)
        c_idx = selectComponent(bkb)
        if c_idx == None:
            return (indices)
        indices.add(c_idx - 1)


def selectState(bkb, c_idx): # Returns state index
    indices = bkb.getAllComponentINodeIndices(c_idx)
    indices.sort()
    print ('States --')
    for i_idx in indices:
        state_name = bkb.getComponentINodeName(c_idx, i_idx)
        print ('\t' + str(i_idx + 1) + '\t---\t' + state_name)
    try:
        choice = int(input('Select state index (or -1 to abort):'))
    except ValueError:
        choice = -1
    if (choice == -1) or (not (choice - 1) in indices):
        choice = None
    del indices
    return (choice)


def selectINode(bkb): # Returns an I-node tuple or None
    print ('Select component:')
    c_idx = selectComponent(bkb)
    if c_idx == None:
        return (None)
    c_idx -= 1
    print ('Select state:')
    i_idx = selectState(bkb, c_idx)
    if i_idx == None:
        return (None)
    i_idx -= 1
    return ( ( c_idx, i_idx ) )


def selectINodes(bkb): # Returns a set of I-node tuples
    inodes = set()
    while True:
        print (len(inodes), end=' ')
        print ('I-nodes have been selected.')
        printINodeTuples(bkb, inodes)
        print ('')
        inode = selectINode(bkb)
        if inode == None:
            return (inodes)
        inodes.add(inode)


def printINodeTuple(bkb, inode_tuple, delim = '\n', prefix = '', no_print = False):
    comp_name = bkb.getComponentName(inode_tuple[0])
    state_name = bkb.getComponentINodeName(inode_tuple[0], inode_tuple[1])
    string = '{} < {} > = < {} >{}'.format(prefix, comp_name, state_name, delim)
    del comp_name
    del state_name
    if not no_print:
        print(string)
    return string

def printINodeTuples(bkb, inode_tuples, delim = '\n', prefix = '', no_print = False):
    string = ''
    for t in inode_tuples:
        string += printINodeTuple(bkb, t, delim, prefix, no_print)
    return string


def printComponents(bkb, components, delim = '\n', no_print = False):
    string = ''
    first = True
    for t_comp in components:
        if not first:
            string += ', '
        else:
            first = False
        comp_name = bkb.getComponentName(t_comp)
        string += '{}'.format(comp_name)
        del comp_name
    string += '{}'.format(delim)
    if not no_print:
        print(string)
    return string


def printInference(bkb, inference, prefix = '', no_print=False):
    string = ''
    string += '{} Probability = {}\n'.format(prefix, inference[0])
    tuples = list(inference[1])
    tuples = sorted(tuples, key=lambda tup: tup[0])
    string += '{} Supported I-Nodes - \n'.format(prefix)
    for inode_tuple in tuples:
        string += printINodeTuple(bkb, inode_tuple, prefix=prefix, no_print=no_print) 
        string += ' --- \n'
    string += '\n'
    tuples = list(inference[4])
    tuples = sorted(tuples, key=lambda tup: tup[0])
    string += '{} Unsupported I-Nodes - \n'.format(prefix)
    for inode_tuple in tuples:
        string += printINodeTuple(bkb, inode_tuple, prefix=prefix, no_print=no_print) 
        string += ' --- \n'
    string += '\n'


# This function sets up the necessary cache tables for reasoning
#     if the BKB was modified.
def setup(bkb):
    logger_std.info('Beginning setup.')
    if not bkb._modified:
        return    

    start_time = time.time()
    bkb.__BKBR_all_snodes = set()
    bkb.__BKBR_snodes = list(bkb._S_nodes)
    bkb.__BKBR_inodes_in_component = {}
    bkb.__BKBR_snodes_by_head = {}
    bkb.__BKBR_snodes_by_tail = {}
    bkb.__BKBR_snodes_incompatible = {}
    bkb.__BKBR_snodes_by_inodes = {}
    c_indices = bkb.getAllComponentIndices()
    bkb.__BKBR_components = set(c_indices)
    #for c_idx in tqdm.tqdm(c_indices, desc='Processing components', leave=False):
    for c_idx in c_indices:
        i_indices = bkb.getAllComponentINodeIndices(c_idx)
        bkb.__BKBR_inodes_in_component[c_idx] = set()
        for i_idx in i_indices:
            inode_tuple = ( c_idx, i_idx )
            bkb.__BKBR_inodes_in_component[c_idx].add(inode_tuple)
            bkb.__BKBR_snodes_by_head[inode_tuple] = set()
            bkb.__BKBR_snodes_by_tail[inode_tuple] = set()
            bkb.__BKBR_snodes_incompatible[inode_tuple] = set()
        del i_indices
    del c_indices
    
    #for s_idx, snode in tqdm.tqdm(enumerate(bkb.__BKBR_snodes), total=len(bkb.__BKBR_snodes), desc='Setup processing S-Nodes'):
    for s_idx, snode in enumerate(bkb.__BKBR_snodes):
        if snode.head == -1: # Skip, not used in reasoning
            continue
        if snode.probability == -1 or snode.probability == 0: # Skip
            continue

        if Debug:
            snode.output(bkb)
        bkb.__BKBR_all_snodes.add(s_idx)
        ( head_c_idx, head_i_idx ) = snode.head
        bkb.__BKBR_snodes_by_head[snode.head].add(s_idx)

        # Add for quick I-Node match to S-Node
        snode_set = set(snode.tail)
        snode_set.add(snode.head)
        snode_set = frozenset(snode_set)
        try:
            bkb.__BKBR_snodes_by_inodes[snode_set]
            print ("BKBReasoning::setup(bkb): Mutual exclusion problem --")
            print ("First S-Node --")
            snode.output(bkb)
            print ("Second S-Node --")
            (bkb.__BKBR_snodes_by_inodes[snode_set]).output(bkb)
            sys.exit(-1)
        except KeyError:
            bkb.__BKBR_snodes_by_inodes[snode_set] = s_idx
            

        # Add to incompatibility tables
        for _, inode_i_idx in bkb.__BKBR_inodes_in_component[head_c_idx]:
            if inode_i_idx == head_i_idx:
                continue
            inode_tuple = ( head_c_idx, inode_i_idx )
            bkb.__BKBR_snodes_incompatible[inode_tuple].add(s_idx)

        for _tidx in range(snode.getNumberTail()):
            tail = snode.getTail(_tidx)
            ( tail_c_idx, tail_i_idx ) = tail
            bkb.__BKBR_snodes_by_tail[tail].add(s_idx)
            
            for _, inode_i_idx in bkb.__BKBR_inodes_in_component[tail_c_idx]:
                if inode_i_idx == tail_i_idx:
                    continue
                inode_tuple = ( tail_c_idx, inode_i_idx )
                bkb.__BKBR_snodes_incompatible[inode_tuple].add(s_idx)

    bkb._modified = False
    logger_std.info("Setup elapsed time = " + str(time.time() - start_time))


# Compute the joint probability and also returns the contributions of all
#   S-nodes
# Sensitive to hosts being set for distributed computing
def computeJoint(bkb, target_inodes, target_components, PartialsCheck = False, num_processes_per_host = 0, hosts_filename = None, mblp_exe_fn = None, maxtime = 10000, license_fn = None, num_solns = 1, updateBKB = True, venv=0, timeout=None):
    # target_inodes and target_components are sets of tuple indices
    # updateBKB == True means it needs to be send to the master host as 
    #   applicable. False means that no need to update.
    # hosts_filename is set for distributed computing solver
    #   -- if set, then num_processes_per_host must also be set
    #   -- this is tested first
    # mblp_exe_fn is the filename to use the MBLP formulation through BARON
    #   -- if set, then the following are required
    #       -- maxtime -- max time to be allocated
    #       -- num_solns -- number of solutions to be generated from best down
    #       -- license_fn -- BARON license
    #   -- this is tested second
    # Returns: {Note -- the hash is a frozenset of I-nodes that includes
    # the target_inodes plus one I-Node each from target_components. 
    # I-nodes are tuples of the form ( component index, I-Node index ) }:
    #   - a probability dictionary hashed on joint instantiation of I-nodes
    #   - a constribution dictionary hashed on S-nodes
    #   - a completed inferences hased on joint instantiation
    #   - count of number of partial inferences explored
    #   - elapsed time not including one-time setup per BKB
    #
    #   or returns None if unable to compute

    if hosts_filename != None:
        setupDistributedReasoning(num_processes_per_host, hosts_filename, venv=venv)
        if updateBKB:
            #-- Setup Snode Hashtable before sending BKB (will help with contributions)
            bkb.__BKBR_snodes = list(bkb._S_nodes)
            if logger_socket.level == 10:
                logger_socket.debug('Parent: sending BKB.')
            sendSocketMessage('New BKB')
            sendSocketMessage(bkb)
        if logger_socket.level == 10:
            logger_socket.debug('Parent: sending target I-Nodes')
        sendSocketMessage('New Target I-Nodes')
        if logger_socket.level == 10:
            logger_socket.debug('Parent: Target I-Nodes {}'.format(target_inodes))
        sendSocketMessage(target_inodes)
        if logger_socket.level == 10:
            logger_socket.debug('Parent: sending target components')
        sendSocketMessage('New Target Components')
        if logger_socket.level == 10:
            logger_socket.debug('Parent: Target components {}'.format(target_components))
        sendSocketMessage(target_components)
        if PartialsCheck:
            if logger_socket.level == 10:
                logger_socket.debug('Parent: setting PartialsCheck')
            sendSocketMessage('Set PartialsCheck')
        else:
            if logger_socket.level == 10:
                logger_socket.debug('Parent: unsetting PartialsCheck')
            sendSocketMessage('Unset PartialsCheck')
        if logger_socket.level == 10:
            logger_socket.debug('Parent: sending Start command.')
        sendSocketMessage('Start')

        if logger_socket.level == 10:
            logger_socket.debug('Parent: waiting for answer.') 
        answer = receiveSocketMessage()
        if logger_std.level == 10:
            logger_std.debug('Parent: answer is {}',format(answer))
        if logger_socket.level == 10:
            logger_socket.debug('Parent: received answer.')
    elif mblp_exe_fn != None:
        answer = BKBR_MBLP._computeJoint(bkb, target_inodes, target_components, "/tmp/" + str(uuid.uuid4()), mblp_exe_fn, maxtime, license_fn, num_solns)
    else:
        answer = _computeJoint(bkb, target_inodes, target_components, PartialsCheck, timeout=timeout)
    return (answer)


# Compute the joint probability and also returns the contributions of all
#   S-nodes
def _computeJoint(bkb, target_inodes, target_components, PartialsCheck = False, timeout=None):
    # target_inodes and target_components are sets of tuple indices
    # Returns: {Note -- the hash is a frozenset of I-nodes that includes
    # the target_inodes plus one I-Node each from target_components. 
    # I-nodes are tuples of the form ( component index, I-Node index ) }:
    #   - a probability dictionary hashed on joint instantiation of I-nodes
    #   - a constribution dictionary hashed on S-nodes
    #   - a completed inferences hased on joint instantiation
    #   - count of number of partial inferences explored
    #   - elapsed time not including one-time setup per BKB
    #
    #   or returns None if unable to compute

    # Setup up caches
    setup(bkb)

    start_time = time.time()
    # Check to make sure there are no imcompatible target_inodes
    common = set()
    assigned_components = set()
    for inode_tuple in target_inodes:
        c_idx = inode_tuple[0]
        assigned_components.add(c_idx)
        common = bkb.__BKBR_inodes_in_component[c_idx] & target_inodes
        if len(common) > 1: # Another I-node found
            if logger_std.level == 10:
                logger_std.debug('Conflicting I-nodes found in target_inodes:')
            while len(common) > 0:
                i_tuple = common.pop()
                if logger_std.level == 10:
                    logger_std.debug('\t < ' + bkb.getComponentName(i_tuple[0]) + ' > = ')
                    logger_std.debug('\t < ' + bkb.getComponentINodeName(i_tuple[0], i_tuple[1]) + ' > = ')
            return None

    if logger_std.level == 10:
        logger_std.debug('Eliminating S-Nodes incompatible with target I-Nodes...')
    compatible_snodes = copy.copy(bkb.__BKBR_all_snodes)
    for inode_tuple in target_inodes:
        compatible_snodes -= bkb.__BKBR_snodes_incompatible[inode_tuple]
    if logger_std.level == 10:
        logger_std.debug('\tremaining #' + str(len(compatible_snodes)))
        logger_std.debug('\t...elapsed time = {} '.format(time.time() - start_time))


    __BKBR_snode_tail = {}
    __BKBR_snodes_by_head = {} 
    __BKBR_snodes_by_tail = {} 


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
    if PartialsCheck:
        partials = set()

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
    #
    #   set of I-nodes supported implies it is supported by some S-nodes used
    start = [ 1.0, set(), list(), copy.copy(assigned_components), copy.copy(target_inodes), copy.copy(fixed_target_components) ]

    queue = list()
    queue.append(start)

    #-- Instantiate Progress Spinner
    if logger_std.level == 10:
        pass
        #spinner = Spinner('Reasoning ')

    while len(queue) > 0:
        #-- Update Spinner
        if logger_std.level == 10:
            pass
            #spinner.next()

        # If we've reached timeout return answers found so far
        if timeout:
            if time.time() - start_time > timeout:
                return ( [ probabilities, contributions, completed_inferences, partials_explored, time.time() - start_time ] )

        item = queue.pop(0)
        partials_explored += 1
        if logger_std.level == 10:
            logger_std.debug('----------------------- Partial #' + str(partials_explored))
            logger_std.debug(printInference(bkb, item, no_print=True))
        if PartialsCheck:
            partial = ( frozenset(item[__idx_supported_inodes]), frozenset(item[__idx_unsupported_inodes]) )
            if len(item[__idx_unsupported_inodes]) > 0 and partial in partials:
                logger_std.error('Partial has already been explored!')
                logger_std.info('Supported --')
                logger_std.info(printINodeTuples (bkb, partial[0], no_print=True))
                logger_std.info('Unsupported --')
                logger_std.info(printINodeTuples (bkb, partial[1], no_print=True))
                sys.exit(-1)
            else:
                partials.add(partial)

        if len(item[__idx_unsupported_inodes]) == 0 and len(item[__idx_unassigned_components]) == 0:
            # Process as finished
            if logger_std.level == 10:
                logger_std.debug('=========Inference built.')
            answer = frozenset(item[__idx_supported_inodes] & target_inodes_filter)
            try:
                probabilities[answer] += item[__idx_probability]
            except KeyError:
                probabilities[answer] = item[__idx_probability]
                completed_inferences[answer] = list()
                contributions[answer] = {}
            item[__idx_snodes_used] = flatten(item[__idx_snodes_used])
            completed_inferences[answer].append(item)
            for s_idx in item[__idx_snodes_used]:
                try:
                    contributions[answer][s_idx] += item[__idx_probability]
                except KeyError:
                    contributions[answer][s_idx] = item[__idx_probability]

        else: # Continue expanding and do branches
            new_open_inodes = copy.copy(item[__idx_unsupported_inodes])
            # select an open I-node if any and requeue possible s-nodes
            if len(new_open_inodes) > 0:
                inode_tuple = new_open_inodes.pop()
                if logger_std.level == 10:
                    logger_std.debug('======Processing I-node ')
                    logger_std.debug(printINodeTuple(bkb, inode_tuple, no_print=True))

                try:
                    candidate_snodes = __BKBR_snodes_by_head[inode_tuple]
                except KeyError: # Make the local reduced version
                    __BKBR_snodes_by_head[inode_tuple] = bkb.__BKBR_snodes_by_head[inode_tuple] & compatible_snodes
                    candidate_snodes = __BKBR_snodes_by_head[inode_tuple]
                if logger_std.level == 10:
                    logger_std.debug('=====candidate snodes ')
                    logger_std.debug('{}'.format(candidate_snodes))
                for s_idx in candidate_snodes: # Build branches
                    if Debug:
                        bkb.__BKBR_snodes[s_idx].output(bkb)

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
                    if logger_std.level == 10:
                        logger_std.debug('+++++++++++Spawning for S-node index = ' + str(s_idx))
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
                    if logger_std.level == 10:
                        logger_std.debug('+++++++++++Spawning for I-node = ')
                        logger_std.debug(printINodeTuple(bkb, inode_tuple, no_print=True))
                    new_item = [ item[__idx_probability], item[__idx_supported_inodes], item[__idx_snodes_used], new_assigned_components, new_inodes, new_open_components ]
                    queue.insert(0, new_item)
                else:
                    new_item = [ item[__idx_probability], item[__idx_supported_inodes], item[__idx_snodes_used], new_assigned_components, set(), new_open_components ]
                    queue.insert(0, new_item)

    if logger_std.level <= 20 and logger_std.level > 0:
        logger_std.info('Statistics:')
        logger_std.info('Total Partials Explored = ' + str(partials_explored))
        for key, value in completed_inferences.items():
            logger_std.info('Answer - ' + str(probabilities[key]))
            for inode_tuple in key:
                logger_std.info(printINodeTuple(bkb, inode_tuple, prefix='', no_print=True))
                logger_std.info(' --- ')
            logger_std.info('Number of inferences = {}'.format(len(value)))


    if PartialsCheck:
        del partials
    del common
    del compatible_snodes
    del __BKBR_snode_tail
    del __BKBR_snodes_by_head
    del __BKBR_snodes_by_tail
    return ( [ probabilities, contributions, completed_inferences, partials_explored, time.time() - start_time ] )


# Takes a list/set of I-nodes as evidence and a target component
#   - evidence_inodes are a set/list of tuples (c_idx, i_idx)
#   - target_component is a component index
# Computes the joint of the evidence with each target component instantiation
# Returns the following tuple:
#   - joint probability for each target I-node in the form
#       prob[target I-Node tuple]
#   - constributions for each target I-node  in the form
#       contributions[target I-Node tuple][S-Node index]
#   - single list of completed inferences 
#       completed_inferfences[inode_tuple]
#   - single value for number of partials explored
#       partials_explored[inode_tuple]
#   - single elapsed time value
#       elapsed_time[inode_tuple]
def update(bkb, evidence_inodes, target_component, PartialsCheck = False, num_processes_per_host = 0, hosts_filename = None, mblp_exe_filename = None, maxtime = 10000, license_fn = "baronlice.txt", num_solns = 1, updateBKB = True, venv=0, timeout=None):
    target_i_indices = bkb.getAllComponentINodeIndices(target_component)
    target_inodes = set([ ( target_component, i_idx ) for i_idx in target_i_indices ])
    evidence_inodes = set(evidence_inodes)
    updates = {}
    contributions = {}
    completed_inferences = {}
    partials_explored = {}
    elapsed_time = {}
    first_pass = updateBKB
#    for inode_tuple in tqdm.tqdm(target_inodes,total=len(target_inodes),desc='Computing I-Node probability', leave = False):
    for inode_tuple in target_inodes:
        inodes = copy.deepcopy(evidence_inodes)
        inodes.add(inode_tuple)
        answer = computeJoint(bkb, inodes, list(), PartialsCheck, num_processes_per_host, hosts_filename, mblp_exe_filename, maxtime, license_fn, num_solns, updateBKB = first_pass, venv=venv, timeout=timeout)
        first_pass = False
        if answer == None:
            updates[inode_tuple] = -1
            contributions[inode_tuple] = None
            completed_inferences[inode_tuple] = list()
            partials_explored[inode_tuple] = 0
            elapsed_time[inode_tuple] = 0
        else:
            try: 
                updates[inode_tuple] = answer[0][frozenset(inodes)]
                contributions[inode_tuple] = answer[1][frozenset(inodes)]
                answer[1] = None
                completed_inferences[inode_tuple] = answer[2] # Preserve
                answer[2] = None
                partials_explored[inode_tuple] = answer[3]
                elapsed_time[inode_tuple] = answer[4]
            except KeyError: # No answer
                updates[inode_tuple] = -1
                contributions[inode_tuple] = None
                completed_inferences[inode_tuple] = list()
                partials_explored[inode_tuple] = 0
                elapsed_time[inode_tuple] = 0
        del answer
            
        if logger_std.level == 10:
            logger_std.debug(printINodeTuple(bkb, inode_tuple, no_print=True))
            logger_std.debug('\t = ' + str(updates[inode_tuple]))
        del inodes
    del target_i_indices
    del target_inodes
    del evidence_inodes
    return (( updates, contributions, completed_inferences, partials_explored, elapsed_time ))

def revise(bkb, evidence_inodes, PartialsCheck=False, num_processes_per_host = 0, hosts_filename = None, mblp_exe_filename = None, maxtime = 10000, license_fn = "baronlice.txt", num_solns = 1, updateBKB = True, venv=0):
    evidence_inodes = set(evidence_inodes)
    return computeJoint(bkb, evidence_inodes, list(), PartialsCheck, num_processes_per_host, hosts_filename, mblp_exe_filename, maxtime, license_fn, num_solns, updateBKB, venv)
 
# See above in update for similar return format
def updateComponents(bkb, evidence_inodes, target_components, PartialsCheck=False, num_processes_per_host = 0, hosts_filename = None, mblp_exe_filename = None, maxtime = 10000, license_fn = "baronlice.txt", num_solns = 1, updateBKB = True, venv=0, timeout=None):
    updates = {}
    contributions = {}
    completed_inferences = {}
    partials_explored = {}
    elapsed_time = {}
    if target_components == None:
        target_components = list()
    if len(target_components) == 0:
        target_component_indices = bkb.getAllComponentIndices()
    else:
        target_component_indices = copy.deepcopy(target_components)
#    for c_idx in tqdm.tqdm(target_component_indices,total=len(target_component_indices),desc='Computing component probabilities'):
    for c_idx in target_component_indices:
        _update = update(bkb, evidence_inodes, c_idx, PartialsCheck, num_processes_per_host, hosts_filename, mblp_exe_filename, maxtime, license_fn, num_solns, updateBKB = updateBKB, venv=venv, timeout=timeout)
        updateBKB = False
        updates.update(_update[0])
        contributions.update(_update[1])
        completed_inferences.update(_update[2])
        partials_explored.update(_update[3])
        elapsed_time.update(_update[4])
        del _update
    del target_component_indices
    return (( updates, contributions, completed_inferences, partials_explored, elapsed_time ))


def isTailCompatible(bkb, s_idx1, s_idx2):
    setup(bkb) # Prepare tables
    # Returns True if tail compatible otherwise False
    #   Assumes setup(bkb) has been called
    tail1 = bkb.__BKBR_snodes[s_idx1].tail
    tail2 = bkb.__BKBR_snodes[s_idx2].tail
    if len(tail1) == 0 or len(tail2) == 0:
        return (True)
    if len(tail2) < len(tail1):
        tail = tail1
        tail1 = tail2
        tail2 = tail
        s_idx = s_idx1
        s_idx1 = s_idx2
        s_idx2 = s_idx
    for inode_tuple in tail1:
        if s_idx2 in bkb.__BKBR_snodes_incompatible[inode_tuple]:
            return (False)
    return (True)
    

# Compute the complementary rule sets for a CRS
def computeComplementaryRuleSets(bkb, component_index):
    # Returns a set of maximal complementary rules sets from crs
    comp_rule_sets = set()
    setup(bkb) # Prepare tables

    num_inodes = bkb.getNumberComponentINodes(component_index)
    table = {} # Contains a table according to I-node
    inode_indices = {}
    inode_max_indices = {}
    __BKBR_snodes_by_head = {}
    for _, i_idx in bkb.__BKBR_inodes_in_component[component_index]:
        table[i_idx] = set() # set of set of S-nodes
        inode_tuple = ( component_index, i_idx )
        __BKBR_snodes_by_head[i_idx] = list(bkb.__BKBR_snodes_by_head[inode_tuple])
        inode_max_indices[i_idx] = len(__BKBR_snodes_by_head[i_idx])
        inode_indices[i_idx] = inode_max_indices[i_idx] - 1

    while True:
        # Attempt to build a current complementary rule set
        rs = set()
        logger_std.info('====Trying ')
        logger_std.info('{}'.format(inode_indices))
        for _, i_idx in bkb.__BKBR_inodes_in_component[component_index]:
            if inode_indices[i_idx] == -1:
                continue
            s_idx = __BKBR_snodes_by_head[i_idx][inode_indices[i_idx]]
            ok = True
            for s_idx_p in rs:
                if not isTailCompatible(bkb, s_idx, s_idx_p):
                    ok = False
                    break
            if ok: # Compatible to all prior S-nodes
                rs.add(s_idx)

        logger_std.info('RS = {}'.format(rs))
        if ok and len(rs) > 0: # A valid complementary rule set 

            logger_std.info('Is ok!')
            # Now check if it is maximal
            s_idx = next(iter(rs)) # Randomly pick an S-node in the set
            ok = True
            try:
                for rs_p in table[s_idx]:
                    if rs.issubset(rs_p): # Is not maximal
                        ok = False
                        break
            except KeyError:
                pass

            if ok:
                comp_rule_sets.add(frozenset(rs))
                # Add to the table
                for s_idx in rs:
                    try:
                        table[s_idx].add(frozenset(rs))
                    except KeyError:
                        table[s_idx] = set()
                        table[s_idx].add(frozenset(rs))
            else:
                logger_std.info('Is subset!')

        # Next set
        more = False
        for _, i_idx in bkb.__BKBR_inodes_in_component[component_index]:
            if inode_indices[i_idx] == -1:
                inode_indices[i_idx] = inode_max_indices[i_idx] - 1
                continue
            inode_indices[i_idx] -= 1
            more = True
            break
        if not more:
            break # Done

    return (comp_rule_sets)


def reason():
    global DISTRIBUTED_SETUP
    PartialsCheck = False
    bkb = BKB.bayesianKnowledgeBase()
    bkb_name = None
    target_inodes = set()
    target_components = set()
    hosts_list = None
    hosts_filename = None
    num_processes_per_host = None
    mblp_exe_filename = None
    maxtime = 10000
    num_solns = 1
    license_fn = "baronlice.txt"

    while True:
        print ('BKB set to \"' + str(bkb_name) + '\"')
        print ('Target I-nodes = ', end='')
        for t in target_inodes:
            print ('\t', end='')
            printINodeTuple(bkb, t)
        print ('Target components = ', end='')
        print ('{ ', end='')
        printComponents(bkb, target_components, delim = '')
        print (' }')
        print ('')
        if PartialsCheck:
            print ('===Partials checking enabled')
        print ('Hosts --')
        print (hosts_list)
        print ('\t# processes per hosts -- {}'.format(num_processes_per_host))
        print ('MBLP_exe_filename -- {}'.format(mblp_exe_filename))
        print ('\tlicense -- {}'.format(license_fn))
        print ('\tmaxtime = {}'.format(maxtime))
        print ('\tnum_solns = {}'.format(num_solns))
        print ('Select action --')
        print ('\t1\t-\tLoad BKB')
        print ('\t5\t-\tCheck Mutex')
        print ('\t10\t-\tAdd target I-Nodes')
        print ('\t15\t-\tReset target I-Nodes')
        print ('\t20\t-\tAdd target components')
        print ('\t25\t-\tReset target components')
        print ('\t30\t-\tLoad pickle evidence')
        print ('\t40\t-\tCompute P(evidence)')
        print ('\t50\t-\tUpdating (uses target I-nodes as evidence')
        print ('\t\t\tand target components to compute posteriors for each')
        print ('\t\t\tif no target components specified, then compute all)')
        print ('\t99\t-\tCompute joint probabilities')
        print ('\t500\t-\tCompute with independent evidence')
        print ('\t999\t-\tCompute complementary rule sets')
        print ('\t1000\t-\tSet partials check')
        print ('\t1001\t-\tUnset partials check')
        print ('\t2000\t-\tLoad in hosts file')
        print ('\t2001\t-\tClear hosts list')
        print ('\t3000\t-\tEnable MBLP solver')
        print ('\t3001\t-\tDisable MBLP solver')
        print ('\t3010\t-\tSet MBLP solver maximum time')
        print ('\t3020\t-\tSet MBLP number of solutions')
        print ('\t5000\t-\tRun BLP solver directory')
        try:
            choice = int(input('Enter you choice (or -1 to exit):'))
        except ValueError:
            continue
        if choice == -1:
            if DISTRIBUTED_SETUP:
                shutdownDistributedReasoning()
            return
        if choice == 1:
            bkb_name = input('Enter bkb filename/path:')
            start = time.time()
            bkb.load(bkb_name)
            print ("...loading elapsed time = " + str(time.time() - start))
            del target_inodes
            target_inodes = set()
            del target_components
            target_components = set()
            continue
        if choice == 5:
            checkMutex(bkb)
            continue
        if choice == 10:
            _target_inodes = selectINodes(bkb)
            target_inodes |= _target_inodes
            continue
        if choice == 15:
            del target_inodes
            target_inodes = set()
            continue
        if choice == 20:
            _target_components = selectComponents(bkb)
            target_components |= _target_components
            continue
        if choice == 25:
            del target_components
            target_components = set()
            continue
        if choice == 30:
            evid_name = input('Enter evidence pickle filename/path:')
            ans = loadPickleEvidence(bkb, evid_name)
            if ans == None:
                print ('No evidence loaded/set -- unchanged.')
            else:
                del target_components
                del target_inodes
                ( target_inodes, target_components ) = ans
            continue
        if choice == 40:
            start_time = time.time()
            if hosts_list == None:
                answer_evid = computeJoint(bkb, target_inodes, set(), PartialsCheck, 0, None)
            else:
                answer_evid = computeJoint(bkb, target_inodes, set(), PartialsCheck, num_processes_per_host, hosts_filename)
            print ('\t...P(E) = ', end='')
            f_target_inodes = frozenset(target_inodes)
            print (answer_evid[0][f_target_inodes])
            del f_target_inodes
            continue
        if choice == 50:
            start_time = time.time()
            answer = updateComponents(bkb, target_inodes, target_components, PartialsCheck, num_processes_per_host, hosts_filename, mblp_exe_filename, maxtime, license_fn, num_solns)
            print ('\t...P(Component,E) =')
            if len(target_components) == 0:
                component_indices = bkb.getAllComponentIndices()
            else:
                component_indices = copy.deepcopy(target_components)
            for c_idx in component_indices:
                inode_indices = bkb.getAllComponentINodeIndices(c_idx)
                comp_name = bkb.getComponentName(c_idx)
                print ('Component - ' + comp_name, end='')
                for i_idx in inode_indices:
                    inode_name = bkb.getComponentINodeName(c_idx, i_idx)
                    inode_tuple = ( c_idx, i_idx )
                    prob = answer[0][inode_tuple]
                    ela = answer[4][inode_tuple]
                    par = answer[3][inode_tuple]
                    print (' [ ' + inode_name + ' = ', end='')
                    print (prob, end='')
                    print (' ] +++ ', end='')
                    print (ela, end='')
                    print (' ( partials = ', end='')
                    print (par, end='')
                    print (' )')
                    del inode_name
                print ('')
                del inode_indices
                del comp_name
            del component_indices
            print ('...updating total elapsed time = ' + str(time.time() - start_time))
            continue
        if choice == 500:
            answers = list()
            for inode_tuple in target_inodes:
                _inodes = list()
                _inodes.append(inode_tuple)
                if hosts_list == None:
                    answers.append(updateComponents(bkb, _inodes, target_components, PartialsCheck, 0, None))
                else:
                    answers.append(updateComponents(bkb, _inodes, target_components, PartialsCheck, num_processes_per_host, hosts_filename))
            print ('...secondary answer --')
            update = secondaryProbability(bkb, target_inodes, target_components, answers)
            for key in sorted(update):
                printINodeTuple(bkb, key, delim=' = ')
                print (update[key])
            continue
        if choice == 99:
            start = time.time()
            if hosts_list == None:
                results = computeJoint(bkb, target_inodes, target_components, PartialsCheck, 0, None)
            else:
                results = computeJoint(bkb, target_inodes, target_components, PartialsCheck, num_processes_per_host, hosts_filename)
            for idx, inference in enumerate(results[2]):
                print ('Inference #' + str(idx) + ': ', end='')
                printINodeTuples(bkb, inference, '')
                print ('')
            print (results[0])
            print ("...reasoning elapsed time = " + str(time.time() - start))
            
            continue
        if choice == 999:
            c_idx = selectComponent(bkb)
            if c_idx == -1:
                continue
            c_idx -= 1
            print ('Complementary rule sets --')
            print (computeComplementaryRuleSets(bkb, c_idx))
            continue
        if choice == 1000:
            PartialsCheck = True
            continue
        if choice == 1001:
            PartialsCheck = False
            continue
        if choice == 2000:
            if DISTRIBUTED_SETUP:
                print ('Shutting down current distributed cluster if appropriate.')
                shutdownDistributedReasoning()
            hosts_filename = input('Enter hosts filename/path:')
            try:
                with open(hosts_filename, 'r') as fn:
                    hosts_list = [ str(line[:-1]) for line in fn.readlines() ]
                fn.close()
            except FileNotFoundError:
                pass
            try:
                num_processes_per_host = int(input('Enter number of processes per hosts or \'enter\' to default to 4:'))
            except ValueError:
                num_processes_per_host = 4
            continue
        if choice == 2001:
            hosts_list = None
            hosts_filename = None
            continue
        if choice == 3000:
            mblp_exe_filename = './baron'
            continue
        if choice == 3001:
            mblp_exe_filename = None
            continue
        if choice == 3010:
            try:
                maxtime = int(input('Enter maximum time in seconds of \'enter\' to default to 10000:'))
            except ValueError:
                maxtime = 10000
            continue
        if choice == 3020:
            try:
                num_solns = int(input("Number of solutions or \'enter\' to default to 1:"))
            except ValueError:
                num_solns = 1
        if choice == 5000:
            try:
                num_solns = int(input("Number of solutions or \'enter\' to default to 1:"))
            except ValueError:
                num_solns = 1
            BKBR_MBLP.solveBKB(bkb, target_components, target_inodes, "test", "BARON", "./baron", 100000, "./baronlice.txt", num_solns)
            continue


def checkMutex(bkb): # Returns True if passes test, False otherwise
    # Will generate a report of mutex issues
    setup(bkb)
    start_time = time.time()
    flag = True
    component_indices = bkb.getAllComponentIndices()
    for cidx in tqdm.tqdm(component_indices, total = len(component_indices), desc='Component testing'):
        inode_indices = bkb.getAllComponentINodeIndices(cidx)
        for iidx in tqdm.tqdm(inode_indices, total = len(inode_indices), desc='I-Node testing', leave=False):
            snode_indices = list(bkb.__BKBR_snodes_by_head[( cidx, iidx )])
            snode_tail = {}
            for sidx1 in tqdm.trange(len(snode_indices), desc='S-Node check', leave=False):
                try:
                    snode_tail[sidx1]
                except KeyError:
                    snode_tail[sidx1] = {}
                    for inode_tuple in bkb.__BKBR_snodes[snode_indices[sidx1]].tail:
                        snode_tail[sidx1][inode_tuple[0]] = inode_tuple[1]
                for sidx2 in tqdm.tqdm(range(sidx1+1, len(snode_indices)), total = len(snode_indices) - sidx1 - 1, desc='...', leave=False):
                    try:
                        snode_tail[sidx2]
                    except KeyError:
                        snode_tail[sidx2] = {}
                        for inode_tuple in bkb.__BKBR_snodes[snode_indices[sidx2]].tail:
                            snode_tail[sidx2][inode_tuple[0]] = inode_tuple[1]
                            
                    if len(bkb.__BKBR_snodes[sidx1].tail) < len(bkb.__BKBR_snodes[sidx2].tail):
                        t1 = sidx1
                        t2 = sidx2
                    else:
                        t1 = sidx2
                        t2 = sidx1
                    passed = False
                    for _cidx, _iidx in snode_tail[t1].items():
                        try:
                            if _iidx != snode_tail[t2][_cidx]:
                                passed = True
                                break
                        except KeyError:
                            pass
                    if not passed:
                        logger_std.warning('-------------------------------------------------------------------')
                        logger_std.warning('S-Node #' + str(sidx1))
                        bkb.__BKBR_snodes[snode_indices[sidx1]].output(bkb)
                        logger_std.warning('\t+++ not mutually exclusive with +++')
                        logger_std.warning('S-Node #' + str(sidx2))
                        bkb.__BKBR_snodes[snode_indices[sidx2]].output(bkb)
                        logger_std.warning('-------------------------------------------------------------------')
                        flag = False
            del snode_tail
            del snode_indices
        del inode_indices
    
    del component_indices
    logger_std.info('\t...elapsed time = ' + str(time.time() - start_time))
    return (flag)
    
    
def checkBucketSums(bkb): # Returns True if passes test, False otherwise
    # Will generate a report of bucket sum issues
    start_time = time.time()
    flag = True
    component_indices = bkb.getAllComponentIndices()
    for cidx in tqdm.tqdm(component_indices, total = len(component_indices), desc='Component testing'):
        compatible_snodes = {}
        snode_tails = {}
        inode_indices = list(bkb.getAllComponentINodeIndices(cidx))
        max_buckets = 1
        snode_indices = {}
        for iidx in tqdm.tqdm(inode_indices, total = len(inode_indices), desc='I-Node processing', leave=False):
            snode_indices[iidx] = list(bkb._S_nodes_by_head[(cidx, iidx)])
            max_buckets *= len(snode_indices)

            for sidx in tqdm.tqdm(snode_indices[iidx], total = len(snode_indices), desc='S-Node preprocessing step 1', leave=False):
                compatible_snodes[sidx] = set()
                snode_tails[sidx] = {}
                for inode_tuple in bkb.__BKBR_snodes[sidx].tail:
                    snode_tails[sidx][inode_tuple[0]] = inode_tuple[1]

        num_buckets = len(inode_indices)
        for _iidx1 in tqdm.trange(num_buckets, desc='I-Node processing Step 2', leave=False):
            for _iidx2 in tqdm.tqdm(range(_iidx1 + 1, num_buckets), total = num_buckets - _iidx1 - 1, desc='I-Node processing Step 2b', leave=False):
                for sidx1 in tqdm.tqdm(snode_indices[inode_indices[_iidx1]], total = len(snode_indices[inode_indices[_iidx1]]), desc='S-Node processing step 2', leave=False):
                    for sidx2 in tqdm.tqdm(snode_indices[inode_indices[_iidx2]], len(snode_indices[inode_indices[_iidx2]]), total = len(snode_indices[inode_indices[_iidx2]]), desc='S-Node compatability', leave=False):
                        if len(bkb.__BKBR_snodes[sidx1].tail) > len(bkb.__BKBR_snodes[sidx2].tail):
                            t1 = sidx2
                            t2 = sidx1
                        else:
                            t1 = sidx1
                            t2 = sidx2
                        comp = True
                        for inode_tuple in bkb.__BKBR_snodes[t1].tail:
                            try:
                                if inode_tuple[1] != snode_tails[t2][inode_tuple[0]]:
                                    comp = False
                                    break
                            except KeyError:
                                pass
                        if comp:
                            compatible_snodes[sidx1].add(sidx2)
                            compatible_snodes[sidx2].add(sidx1)

            # Building buckets
            combo = [-1] * num_buckets
            pbar = tqdm.tqdm(total = max_buckets, desc = 'Bucket creation...', leave = False)
            while True:
                value = 0
                for idx in range(num_buckets):
                    # check compatability
                    if combo[idx] == -1:
                        continue
                    
                    value += bkb.__BKBR_snodes[snode_indices[inode_indices[idx]]]
                pbar.update()
            pbar.close()
                
        del snode_indices
        del snode_tails
        del inode_indices
        del compatible_snodes
        
    del component_indices


def checkCompatible(inode_tuples1, inode_tuples2):
# Returns True if there are no conflicting I-Nodes between the two sets
    inode_tuples = list(inode_tuples1 | inode_tuples2)
    c = collections.Counter(elem[0] for elem in inode_tuples)
    if sum(v for k, v in c.items() if v > 1) > 0:
        no_conflict = False
    else:
        no_conflict = True
    del inode_tuples
    del c
    return (no_conflict)


def checkOverlap(inode_tuples1, inode_tuples2):
# Returns True if there is an overlap of components between the two sets
    c1 = set([ x[0] for x in inode_tuples1])
    c2 = set([ x[0] for x in inode_tuples2])
    if len(c1 & c2) > 0:
        overlap = True
    else:
        overlap = False
    del c1
    del c2
    return (overlap)


def buildIncompatibleInferencesPerInference(inferences):
# inferences are a set of sets of inode_tuples
# Returns a dictionary hashed by inference (set of inode_tuples = frozenset) of all incompatible inferences (set of inode_tuples) 
#   each set includes the Inference
    answer = dict()
    _inferences = list()
    for inference in inferences:
        joint = frozenset(inference)
        _inferences.append(joint)
        answer[joint] = set()
        answer[joint].add(joint)
    while len(_inferences) > 0:
        inference = _inferences.pop()
        for inf2 in _inferences:
            if not checkCompatible(inference, inf2):
                answer[inference].add(inf2)
                answer[inf2].add(inference)
    del _inferences
    return (answer)


def extractCompletedInferences(updating_answers):
# updating_answers is a collection of answers from updating, i.e., from 
#   multiple queries to updateComponents(...)
# Returns a tuple (p, infs):
#   - p is a dictionary that hashes an inference to a joint probability
#   - infs is a set of inferences (i.e., sets of I-Node tuples (cidx, iidx))
#   extracted from updating_answers
    global __idx_probability 
    global __idx_supported_inodes 
    p = dict()
    infs = set()
    for answer in tqdm.tqdm(updating_answers, total=len(updating_answers), desc='Extracting completed inferences from answer sets'):
        # answer[2] is the completed inferences dictionary
        for inode_tuple, completed_inferences in tqdm.tqdm(answer[2].items(), total=len(answer[2].items()), desc='Completed inference', leave=False): 
            if len(completed_inferences) == 0:
                continue
            for key, value in tqdm.tqdm(completed_inferences.items(),total=len(completed_inferences.items()),desc='...answers',leave=False):
                for inference in tqdm.tqdm(value,total=len(completed_inferences),desc='...processing inference',leave=False):
                    joint = frozenset(inference[__idx_supported_inodes]) # set of I-node tuples in completed inference
                    p[joint] = inference[__idx_probability] # probability of completed inference
                    infs.add(joint)
    return ( (p, infs) )


def secondaryProbability(bkb, evidence_inodes, target_components, updating_answers):
# Answer is a hash of answers in the format of updateComponents answer
#   but can also be constructed elsewhere
# Returns an aggregate probability for each target component I-Node (dictionary)
    p, infs = extractCompletedInferences(updating_answers)
    if Debug:
        print ('Total number of completed inferences =' + str(len(infs)))
    _update = dict()
    if target_components == None:
        target_components = list()
    if len(target_components) == 0:
        target_component_indices = bkb.getAllComponentIndices()
    else:
        target_component_indices = copy.deepcopy(target_components)

    # Build best sets for each target I-Node
    for c_idx in tqdm.tqdm(target_component_indices,total=len(target_component_indices),desc='Computing secondary component probabilities'):
        target_i_indices = bkb.getAllComponentINodeIndices(c_idx)
        target_inodes = set([ ( c_idx, i_idx ) for i_idx in target_i_indices ])
        for inode_tuple in tqdm.tqdm(target_inodes,total=len(target_inodes),desc='...I-Node probability',leave=False):
            # Extract inferences that contain inode_tuple
            #   - may not be all mutually exclusive
            filtered_infs = set()
            for inference in infs:
                if inode_tuple in inference:
                    filtered_infs.add(inference)
            if Debug:
                print ('==============================')
                printINodeTuple(bkb, inode_tuple)
                print ('')
                print ('\t...inference count = ' + str(len(filtered_infs)))

            incompats = buildIncompatibleInferencesPerInference(filtered_infs)
            # Compute probability sums
            best_prob = 0
            for inference_key, inferences in tqdm.tqdm(incompats.items(),total=len(incompats.items()),desc='......processing incompatible sets',leave=False):
                prob = 0
                for inference in tqdm.tqdm(inferences,total=len(inferences),desc='.........summing probabilities',leave=False):
                    prob += p[inference]
                if prob > best_prob:
                    best_prob = prob
            if Debug:
                print ('Best probability = ' + str(best_prob))
            _update[inode_tuple] = best_prob
            del incompats
            del filtered_infs
        del target_i_indices
        del target_inodes
    del p
    del infs
    del target_component_indices
    if Debug:
        print ('------Secondary answers')
        print (_update)
    return (_update)


def flatten_list(container):
    for i in container:
        if isinstance(i, (list, tuple)):
            yield from flatten_list(i)
        else:
            yield i

def flatten(container):
    generator = flatten_list(container)
    return [item for item in generator]

def loadPickleEvidence(bkb, filename):
    try:
        with open (filename, 'rb') as fn:
            data = (pickle.load(fn))
    except FileNotFoundError:
        return (None)
    target_inodes = set()
    evid_dict = data['evidence']
    for comp, state in evid_dict.items():
        c_idx = bkb.getComponentIndex(comp)
        if c_idx == -1:
            print ('loadPickleEvidence(...) -- Unknown component <{}>!'.format(comp))
            return (None)
        i_idx = bkb.getComponentINodeIndex(c_idx, state)
        if i_idx == -1:
            print ('loadPickleEvidence(...) -- Unknown state <{}> for component <{}>!'.format(comp, state))
            return (None)
        target_inodes.add((c_idx, i_idx))
    target_comp = data['targets']
    target_component = set()
    for comp in target_comp:
        c_idx = bkb.getComponentIndex(comp) 
        if c_idx == -1:
            print ('loadPickleEvidence(...) -- Unknown component <{}>!'.format(comp))
            return (None)
        target_component.add(c_idx)
    return ( ( target_inodes, target_component ) )
