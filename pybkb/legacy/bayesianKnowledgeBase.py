import sys
import copy
import io
import json
import numpy as np
from contextlib import redirect_stdout
from collections import defaultdict
import itertools

# Comment line below to run in C++ branch.
import compress_pickle
import pickle

import pybkb.utils.probability as mi

Debug = False

#-- This is Beta Version.


# Remove empty lines or lines of whitespace from list of lines
def cleanEmptyLines(lines):
    new_lines = list()
    for line in lines:
        if len(line.strip()) == 0:
            continue
        new_lines.append(line)
    return(new_lines)


class BKB_I_node:
    def reset(self):
        del self._name
        del self._component_index
        del self._index

    def __init__(self, init_name = '', init_component_index = -1, init_index = -1):
        self._name = copy.deepcopy(init_name)
        self._component_index = init_component_index
        self._index = init_index

    def load(self, bkb, component_index, state_index, lines, line_start):
        self.reset()
        line_pos = line_start
        line = lines[line_pos].lstrip()
        temp = line.split()
        length = int(temp[0])
        start_idx = line.find(' ')
        self._name = line[start_idx:].lstrip()
        self._name = self._name[:length]
        self._component_index = component_index
        self._index = state_index
        if Debug:
            print ('State found - <' + self._name +'>')
        line_pos += 1
        return(line_pos)

    def save(self, fileobject):
        fileobject.write('\t\t' + str(len(self._name)) + " " + self._name + '\n')
 
    def output(self):
        print ('<' + self._name + '>')    

    def match(self, statename, contains=False):
        if contains:
            return (statename in self._name)
        return (self._name == statename)

    def getName(self):
        return (copy.deepcopy(self._name))

    def getIndex(self):
        return (self._index)

    def getComponentIndex(self):
        return (self._component_index)
               

class BKB_component:
    def reset(self):
        del self._name
        self._name = ''
        del self._states_index
        self._states_index = dict() # maps string to index
        for _, inode in self._states.items():
            del inode
        del self._states
        self._states = dict() # maps index to i-node
        self._index = -1
        self._avail_state_index = 0

    def __init__(self, init_name = '', init_component_index = -1):
        self._name = copy.deepcopy(init_name)
        self._states_index = dict()
        self._states = dict()
        self._index = init_component_index
        self._avail_state_index = 0
   
    def load(self, bkb, component_index, lines, line_start):
        self.reset()
        self._index = component_index
        line_pos = line_start
        line = lines[line_pos].lstrip()
        temp = line.split()
        length = int(temp[0])
        start_idx = line.find(' ')
        self._name = line[start_idx:].lstrip()
        self._name = self._name[:length]
        if Debug:
            print ('Component found - <' + self._name + '>')
        line_pos += 1
        temp = lines[line_pos].split()
        if temp[0] != 'STATES:':
            msg = 'Expected \"STATES:\" instead got <' + temp[0] + '>'
            sys.exit(msg)
        num_states = int(temp[1])
        if Debug:
            print ('# of states = ' + str(num_states))
        line_pos += 1
        for iidx in range(num_states):
            i_node = BKB_I_node()
            line_pos = i_node.load(bkb, self._index, self._avail_state_index, lines, line_pos)
            self._addINode(i_node)
        return (line_pos)

    def save(self, fileobject):
        fileobject.write(str(len(self._name)) + ' ' + self._name + '\n')
        fileobject.write('\tSTATES: ' + str(self.getNumberStates()) + '\n')
        for key, state in self._states.items():
            state.save(fileobject)

    def output(self):
        print ('Component - <' + self._name + '>')
        print ('States (' + str(len(self._states)) + ') - ')
        for key, state in self._states.items():
            state.output()

    def getIndex(self):
        return (self._index)

    def getName(self):
        return (copy.deepcopy(self._name))
    
    def getNumberStates(self):
        return (len(self._states))
        
    def findState(self, state_name, contains=False): # Returns an index
        if contains:
            for name, idx in self._states_index.items():
                if state_name in name:
                    return (idx)
        try:
            return (self._states_index[state_name])
        except KeyError:
            return (-1)

    def getAllStateIndices(self):
        indices = list()
        for _, idx in self._states_index.items():
            indices.append(idx)
        return (indices)

    def getStateName(self, index):
        try:
            return (self._states[index].getName())
        except KeyError:
            return ('')

    def getAllStateNames(self): # Retuns set of state names
        names = set()
        for name, _ in self._states_index.items():
            names.add(copy.deepcopy(name))
        return (names)

    def addState(self, state_name): # Adds state_name if new and returns idx
        idx = self.findState(state_name)
        if idx == -1: # New I-node
            inode = BKB_I_node(state_name, self._index, self._avail_state_index)
            self._states_index[state_name] = self._avail_state_index
            self._states[self._avail_state_index] = inode
            idx = self._avail_state_index
            self._avail_state_index += 1
        return (idx)

    def removeState(self, state_index): # Returns true if successfully removed
        try:
            inode = self._states[state_index]
            del self._states_index[inode.getName()]
            del self._states[state_index]
            del inode
            return (True)
        except KeyError:
            return (False)

    def match(self, compname, contains=False):
        if contains:
            return compname in self._name
        return (self._name == compname)

# Private functions
           
    def _getStateIndex(self, inode): # Private function
        if inode.getComponentIndex() != self:
            return (-1)
        try:
            return (self.states_index[inode.getName()])
        except KeyError:
            return (-1)

    def _getState(self, index): # Returns the I-node or -1
        try:
            return (self._states[index])
        except KeyError:
            return (-1)

    def _addINode(self, inode): # Adds I-node if new
        name = inode.getName()
        if self.findState(name) == -1: # New I-node
            if not inode.getIndex() == self._avail_state_index:
                print ('BKB_component::addINode(inode) -- available state index does not match I-node\'s index')
                sys.exit(msg)
            self._states_index[name] = self._avail_state_index
            self._states[self._avail_state_index] = inode
            self._avail_state_index += 1


class BKB_S_node:
    def reset(self):
        del self.head
        self.head = -1
        del self.tail
        self.tail = list()
        self.probability = -1

    def __init__(self, init_component_index = -1, init_state_index = -1, init_probability = -1, init_tail = list()): # Only a shallow copy of tail is made
        if init_component_index == -1 or init_state_index == -1:
            self.head = -1
        else:
            self.head = ( init_component_index, init_state_index )
        self.probability = init_probability
        self.tail = init_tail # only shallow copy wanted

    def load(self, bkb, component_index, i_node_index, lines, line_start):
        self.reset()
        self.head = ( component_index, i_node_index )
        line_pos = line_start
        temp = lines[line_pos].split()
        line_pos += 1
        self.probability = float(temp[0])
        if temp[1] != 'TAIL:':
            msg = 'Expected \"TAIL:\" instead retrieved <' + temp[1] + '>'
            sys.exit(msg)
        num_tails = int(temp[2])
        for tidx in range(num_tails):
            line = lines[line_pos].lstrip()
            line_pos += 1
            temp = line.split()
            length = int(temp[0])
            start_idx = line.find(' ')
            line = line[start_idx:].lstrip()
            compname = line[:length]
            component_index = bkb.getComponentIndex(compname)
            if component_index == -1:
                msg = 'Unknown component <' + compname + '>'
                sys.exit(msg)
            line = line[length:].lstrip()
            temp = line.split()
            length = int(temp[0])
            start_idx = line.find(' ')
            line = line[start_idx:].lstrip()
            statename = line[:length]
            i_node_index = bkb.getComponentINodeIndex(component_index, statename)
            if i_node_index == -1:
                msg == 'Unknown state <' + statename + '> for component <' + compname + '>'
                sys.exit(msg)
            self.tail.append( ( component_index, i_node_index ) )
        if Debug:
            print ('Found S-node:')
            self.output(bkb)
        return(line_pos)

    def output(self, bkb):
        for i_node in self.tail:
            comp_index = i_node[0]
            comp = bkb._getComponent(comp_index)
            inode_index = i_node[1]
            inode = bkb._getComponentINode(comp_index, inode_index)
            print ('<' + comp.getName() + '> = ', end='')
            print ('<' + inode.getName() + '>', end=' ')
        print ('\t( ' + str(self.probability) + ' ) ==> ', end='')
        if self.head != -1:
            comp_index = self.head[0]
            comp = bkb._getComponent(comp_index)
            inode_index = self.head[1]
            inode = bkb._getComponentINode(comp_index, inode_index)
            print ('<' + comp.getName() + '> = ', end='')
            print ('<' + inode.getName() + '>')
        else:
            print ('none')

    def getHead(self):
        return (self.head)

    def getProbability(self):
        return (self.probability)

    def getNumberTail(self):
        return (len(self.tail))

    def getTail(self, index):
        return (self.tail[index])

    def calculate_mdl(self, bkb, node_encoding_len, data, feature_states, feature_states_map, store):
        struct_mdl = (self.getNumberTail() + 1)*node_encoding_len
        # Get head info
        head_comp_idx, head_state_idx = self.getHead()
        head_comp_name = bkb.getComponentName(head_comp_idx)
        head_state_name = bkb.getComponentINodeName(head_comp_idx, head_state_idx)
        x_state_idx = feature_states_map[(head_comp_name, head_state_name)]
        # Get tail info
        parent_state_indices = []
        for i in range(self.getNumberTail()):
            tail_comp_idx, tail_state_idx = self.getTail(i)
            tail_comp_name = bkb.getComponentName(tail_comp_idx)
            tail_state_name = bkb.getComponentINodeName(tail_comp_idx, tail_state_idx)
            parent_state_indices.append(
                    feature_states_map[(tail_comp_name, tail_state_name)]
                    )
        # Calculate data MDL
        data_mdl, store = mi.instantiated_mutual_info(data, x_state_idx, parent_state_indices, store)
        # Number of atomic events represented by an S-node is just 1
        return data_mdl - struct_mdl, store

    def calculate_mdl_ent(self, bkb, node_encoding_len, data, feature_states, feature_states_map, store, only_mdl=None):
        struct_mdl = (self.getNumberTail() + 1)*node_encoding_len
        # Get head info
        head_comp_idx, head_state_idx = self.getHead()
        head_comp_name = bkb.getComponentName(head_comp_idx)
        head_state_name = bkb.getComponentINodeName(head_comp_idx, head_state_idx)
        x_state_idx = feature_states_map[(head_comp_name, head_state_name)]
        # Get tail info
        parent_state_indices = []
        for i in range(self.getNumberTail()):
            tail_comp_idx, tail_state_idx = self.getTail(i)
            tail_comp_name = bkb.getComponentName(tail_comp_idx)
            tail_state_name = bkb.getComponentINodeName(tail_comp_idx, tail_state_idx)
            parent_state_indices.append(
                    feature_states_map[(tail_comp_name, tail_state_name)]
                    )
        # Calculate data MDL
        data_mdl, store = mi.instantiated_entropy(data, x_state_idx, parent_state_indices, store)
        # Number of atomic events represented by an S-node is just 1
        if only_mdl is None:
            return -data_mdl - struct_mdl, store
        if only_mdl == 'data':
            return -data_mdl, store
        if only_mdl == 'model':
            return -struct_mdl, store
        raise ValueError(f'Unknown option {only_mdl}. Must be one of [data, model].')


    def isMutex(self, other_snode, with_head=False):
        if with_head and self.getHead() != other_snode.getHead():
            return True
        if self.tail is None:
            tail1 = []
        else:
            tail1 = copy.deepcopy(self.tail)
        if other_snode.tail is None:
            tail2 = []
        else:
            tail2 = copy.deepcopy(other_snode.tail)
        if len(tail1) == 0 or len(tail2) == 0:
            return False
        if len(tail2) < len(tail1):
            tail = tail1
            tail1 = tail2
            tail2 = tail
        for inode1_tuple in tail1:
            for inode2_tuple in tail2:
                if inode1_tuple[0] == inode2_tuple[0]:
                    if inode1_tuple[1] != inode2_tuple[1]:
                        return True
        return False


    def isCompatible(self, other_snode, with_head=False):
        tail1 = copy.deepcopy(self.tail)
        tail2 = copy.deepcopy(other_snode.tail)
        if with_head:
            tail1.append(self.head)
            tail2.append(other_snode.head)
        if len(tail1) == 0 or len(tail2) == 0:
            return (True)
        if len(tail2) < len(tail1):
            tail = tail1
            tail1 = tail2
            tail2 = tail
        for inode_tuple_1 in tail1:
            for inode_tuple_2 in tail2:
                if inode_tuple_1[0] == inode_tuple_2[0]:
                    if inode_tuple_1[1] != inode_tuple_2[1]:
                        return False
        return True

    def __eq__(self, other):
        if self.head != other.head:
            return False
        if self.probability != other.probability:
            return False
        if len(self.tail) != len(other.tail):
            return False
        for tail in self.tail:
            found = False
            for other_tail in other.tail:
                if tail == other_tail:
                    found = True
                    break
            if not found:
                return False
        return True

    def __hash__(self):
        tail = frozenset(self.tail)
        obj = (self.head, self.probability, tail)
        return hash(obj)

class HumanReadableSnode:
    def __init__(self, probability, head, tail):
        self.probability = probability
        self.head = head
        self.tail = tail

    @classmethod
    def make(cls, snode, bkb):
        probability = snode.getProbability()
        head = (bkb.getComponentName(snode.head[0]),
                bkb.getComponentINodeName(snode.head[0], snode.head[1]))
        tail = []
        for tail_idx in snode.tail:
            tail.append((bkb.getComponentName(tail_idx[0]),
                         bkb.getComponentINodeName(tail_idx[0], tail_idx[1])))
        return cls(probability, head, tail)

    def __eq__(self, other):
        if self.probability == other.probability:
            if self.head == other.head:
                if len(set(self.tail) ^ set(other.tail)) == 0:
                    return True
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        string = ''
        string += 'Probability = {}\nHead = {}\nTail = {}'.format(self.probability,
                                                                  self.head,
                                                                  self.tail)
        return string

class bayesianKnowledgeBase:
    def reset(self):
        del self._components_index
        self._components_index = dict() # maps string to index
        for _, comp in self._components.items():
            del comp
        del self._components
        self._components = dict() # maps index to component
        for snode in self._S_nodes:
            del snode
        del self._S_nodes
        self._S_nodes = set()
        self._modified = True
        del self._src_components
        self._src_components = set() # Set of component indices
        self._avail_component_index = 0
        del self._S_nodes_by_head_tuple
        self._S_nodes_by_head_tuple = dict()
        del self._S_nodes_by_tail_tuple
        self._S_nodes_by_tail_tuple = dict()

    def __init__(self, name='bkb0'):
        self._components_index = dict() # maps string to index
        self._components = dict() # maps index to component
        self._S_nodes = set()
        self._modified = True
        self._src_components = set()
        self._avail_component_index = 0
        self._name = copy.deepcopy(name)
        self._S_nodes_by_head_tuple = dict()
        self._S_nodes_by_tail_tuple = dict()

    def json(self, indent=2):
        _dict = {"I-nodes": defaultdict(list), "S-nodes": []}
        for snode in self._S_nodes:
            snode_dict = {}
            head_comp_idx, head_state_idx = snode.getHead()
            head_comp_name = self.getComponentName(head_comp_idx)
            head_state_name = self.getComponentINodeName(head_comp_idx, head_state_idx)
            snode_dict['HeadComponent'] = head_comp_name
            snode_dict['HeadState'] = head_state_name
            snode_dict["Tail"] = []
            for tail_idx in range(snode.getNumberTail()):
                tail_comp_idx, tail_state_idx = snode.getTail(tail_idx)
                tail_comp_name = self.getComponentName(tail_comp_idx)
                tail_state_name = self.getComponentINodeName(tail_comp_idx, tail_state_idx)
                snode_dict["Tail"].append(
                        {
                            "Component": tail_comp_name,
                            "State": tail_state_name,
                            }
                        )
            snode_dict["Probability"] = snode.getProbability()
            _dict["S-nodes"].append(snode_dict)
        for comp_idx in self.getAllComponentIndices():
            comp_name = self.getComponentName(comp_idx)
            for state_idx in self.getAllComponentINodeIndices(comp_idx):
                state_name = self.getComponentINodeName(comp_idx, state_idx)
                _dict["I-nodes"][comp_name].append(state_name)
        _dict["I-nodes"] = dict(_dict["I-nodes"])
        return json.dumps(_dict, indent=indent)

    def getCausalRuleSet(self, comp_idx=None):
        crs = defaultdict(list)
        for snode in self._S_nodes:
            head_comp, head_state = snode.getHead()
            crs[head_comp].append(snode)
        if comp_idx is None:
            return dict(crs)
        else:
            return crs[comp_idx]

    def getComplementarySets(self):
        # Get max complementary sets
        complementary_sets = []
        for snode in self._S_nodes:
            head_comp, head_state = snode.getHead()
            if len(complementary_sets) == 0:
                complementary_sets.append([snode])
                continue
            processed_snode = False
            for i, no_mutex_snodes in enumerate(complementary_sets):
                # Take the first q cause they'll all be non-mutex.
                no_mutex_snode1 = no_mutex_snodes[0]
                no_mutex_snode_head_comp = no_mutex_snode1.getHead()[0]
                no_mutex_head_states = [_snode.getHead()[1] for _snode in no_mutex_snodes]
                if not snode.isMutex(no_mutex_snode1, with_head=False) and head_comp == no_mutex_snode_head_comp and head_state not in no_mutex_head_states:
                    complementary_sets[i].append(snode)
                    processed_snode = True
                    break
                elif not snode.isMutex(no_mutex_snode1, with_head=False) and head_comp == no_mutex_snode_head_comp and head_state in no_mutex_head_states:
                    print(HumanReadableSnode.make(snode, self))
                    print(HumanReadableSnode.make(no_mutex_snode1, self))
                    raise ValueError('Found compatible S-nodes.')
            if not processed_snode:
                complementary_sets.append([snode])
        return complementary_sets

    def makeGraph(
            self,
            show=True,
            save_file=None,
            dpi=None, 
            layout=None,
            size_multiplier=None,
            ):
        # Comment out line if using in C++ library.
        from .bkb_grapher import make_graph

        graph = make_graph(self)
        fig = graph.draw(
                layout=layout,
                show=show,
                save_file=save_file,
                dpi=dpi,
                size_multiplier=size_multiplier,
                )
        return graph, fig

    def load(self, filename, use_pickle=False, compress=True):
        if use_pickle:
            with open(filename, 'rb') as f_:
                if compress:
                    return compress_pickle.load(f_, compression="lz4")
                else:
                    return pickle.load(f_)
        self.reset()
        try:
            with open(filename, 'r') as fn:
                lines = fn.readlines()
        except FileNotFoundError:
            return
        lines = cleanEmptyLines(lines)
        line_pos = 0

        # First line should be COMPONENTS: XX
        component_line = lines[line_pos].split()
        if component_line[0] != 'COMPONENTS:':
            msg = 'Expecting \"COMPONENTS:\" instead retrieved <' + component_line[0] + '>!'
            sys.exit(msg)
        num_components = int(component_line[1])
        line_pos += 1
        for cidx in range(num_components):
            comp = BKB_component()
            line_pos = comp.load(self, self._avail_component_index, lines, line_pos);
            self._addComponent(comp)

        # Now loading S-nodes
        total_lines = len(lines)
        while line_pos < total_lines:
            line = lines[line_pos].split()
            if line[0] == 'FREE_SUPPORTS:':
                break
            # first two lines is the head
            head = lines[line_pos] # component name
            line_pos += 1
            line = head.lstrip()
            temp = line.split()
            length = int(temp[0])
            start_idx = line.find(' ')
            compname = line[start_idx:].lstrip()
            compname = compname[:length]
            component = self._findComponent(compname)
            if component == -1:
                msg = 'Unknown component <' + compname + '>'
                sys.exit(msg)
            cidx = component.getIndex()
            num_states = component.getNumberStates()
            for idx in range(num_states):
                line = lines[line_pos].lstrip() # state name
                line_pos += 1
                temp = line.split()
                length = int(temp[0])
                start_idx = line.find(' ')
                statename = line[start_idx:].lstrip()
                statename = statename[:length]
                iidx = component.findState(statename)
                if iidx == -1:
                    msg = 'Unknown state <' + statename + '> for component <' + compname + '> for head'
                    sys.exit(msg)
                line = lines[line_pos].split()
                line_pos += 1
                if line[0] != 'SUPPORTS:':
                    msg = 'Expected \"SUPPORTS:\" instead retrieved <' + line[0] + '>!'
                    sys.exit(msg)
                num_snodes = int(line[1])
                for sidx in range(num_snodes):
                    s_node = BKB_S_node()
                    line_pos = s_node.load(self, cidx, iidx, lines, line_pos)
                    self.addSNode(s_node)

    def save(self, filename, use_pickle=False, compress=True):
        if use_pickle:
            with open(filename, 'wb') as f_:
                if compress:
                    compress_pickle.dump(self, f_, compression="lz4")
                else:
                    pickle.dump(self, f_)
            return
        with open(filename, 'w') as fn:
            fn.write('COMPONENTS: ' + str(self.getNumberComponents()) + '\n')
            for _, component in self._components.items():
                component.save(fn)
            for cidx, component in self._components.items():
                fn.write(str(len(component.getName())) + ' ' + component.getName() + '\n')
                inode_indices = component.getAllStateIndices()
                for iidx in inode_indices:
                    state = component._getState(iidx)
                    name = state.getName()
                    try:
                        num_s_nodes = len(self._S_nodes_by_head_tuple[( cidx, iidx)])
                    except KeyError:
                        num_s_nodes = 0
#                        msg = '{} = {} has no S nodes.'.format(self.getComponentName(cidx), self.getComponentINodeName(cidx, iidx))
#                        self.output()
#                        sys.exit(msg)
                    fn.write('\t' + str(len(name)) + ' ' + name + '\n')
                    fn.write('\t\tSUPPORTS: ' + str(num_s_nodes) + '\n')
                    if num_s_nodes == 0:
                        continue
                    for s_node in self._S_nodes_by_head_tuple[( cidx, iidx )]:
                        fn.write('\t\t\t' + str(s_node.getProbability()) + ' TAIL: ' + str(s_node.getNumberTail()) + '\n')
                        for tidx in range(s_node.getNumberTail()):
                            tail = s_node.getTail(tidx)
                            ( comp_index, state_index ) = tail
                            comp_ = self._getComponent(comp_index)
                            state_ = self._getComponentINode(comp_index, state_index)
                            fn.write('\t\t\t\t' + str(len(comp_.getName())) + ' ' + comp_.getName() + ' ' + str(len(state_.getName())) + ' ' + state_.getName() + '\n')
                del inode_indices
            fn.write('FREE_SUPPORTS:' + '\n')

    def output(self):
        print ('# of components = ' + str(self.getNumberComponents()))
        for key, component in self._components.items():
            component.output()
        for s_node in self._S_nodes:
            s_node.output(self)

    def getNumberComponents(self):
        return (len(self._components))

    def getComponentIndex(self, comp_name): # Only looks for the name
        try:
            return (self._components_index[comp_name])
        except KeyError:
            return (-1)

    def getAllComponentIndices(self):
        indices = list()
        for _, idx in self._components_index.items():
            indices.append(idx)
        return (indices)

    def getComponentName(self, index):
        return (copy.deepcopy(self._components[index].getName()))

    def getAllComponentNames(self):
        names = list()
        for name, _ in self._components_index.items():
            names.append(copy.deepcopy(name))
        return (names)

    def addComponent(self, comp_name): # Create new component if possible and returns component index
        try:
            return (self._components_index[comp_name])
        except KeyError: # New component
            pass
        comp = BKB_component(comp_name, self._avail_component_index)
        self._components_index[comp_name] = self._avail_component_index
        self._components[self._avail_component_index] = comp
        self._modified = True
        self._avail_component_index += 1
        return (self._avail_component_index - 1)

    def getNumberComponentINodes(self, component_index):
        try:
            return (self._components[component_index].getNumberStates())
        except KeyError:
            return (-1)

    def getComponentINodeIndex(self, component_index, state_name):
        try: 
            return (self._components[component_index].findState(state_name))
        except KeyError:
            return (-1)

    def getAllComponentINodeIndices(self, component_index):
        try:
            return (self._components[component_index].getAllStateIndices())
        except KeyError:
            return (-1)

    def getComponentINodeName(self, component_index, inode_index):
        try:
            return (self._components[component_index].getStateName(inode_index))
        except KeyError:
            return (-1)
   
    def findINode(self, component_index, statename, contains=False):
        try:
            return(self._components[component_index].findState(statename, contains))
        except KeyError:
            return (-1)

    def getINodeNames(self): # Returns a list of tuples ( component_name, state_name )
        i_nodes = list()
        for _, component in self._components.items():
            names = component.getAllStateNames()
            for name in names:
                i_nodes.append((component.getName(), name))
        return (i_nodes)
    
    def addComponentState(self, component_index, state_name): # Returns index even if found
        iidx = self.getComponentINodeIndex(component_index, state_name)
        if iidx != -1:
            return (iidx) # already found
        iidx = self._components[component_index].addState(state_name)
        try:
            self._S_nodes_by_head_tuple[( component_index, iidx )]
        except KeyError:
            self._S_nodes_by_head_tuple[( component_index, iidx )] = set()
            self._S_nodes_by_tail_tuple[( component_index, iidx )] = set()
        self._modified = True
        return (iidx)

    def removeComponent(self, component_index): # Returns True if successful
        try:    
            component = self._components[component_index]
            inode_indices = component.getAllStateIndices()
            for iidx in inode_indices:
                self.removeComponentState(component_index, iidx)
            del inode_indices
            del self._components_index[component.getName()]
            del self._components[component_index]
            del component
            self._modified = True
            return (True)
        except KeyError:
            return (False)

    def removeComponentState(self, component_index, state_index): # Returns True if successful
        try:
            component = self._components[component_index]
            inode_tuple = (component_index, state_index)
            # Find S-Nodes involving this I-Node
            snodes = self._S_nodes_by_head_tuple[inode_tuple] | self._S_nodes_by_tail_tuple[inode_tuple]
            for snode in snodes:
                self.removeSNode(snode)
            del self._S_nodes_by_head_tuple[( component_index, state_index )]
            del self._S_nodes_by_tail_tuple[( component_index, state_index )]
                
            if not component.removeState(state_index):
                return (False)
            self._modified = True
            return (True)
        except KeyError:
            return (False)


    def removeSNode(self, s_node):
        # NOTE -- s_node is only remoced from the BKB but not deallocated.
        #input('Result of remove Snode = {}'.format(s_node in self._S_nodes_by_head_tuple[s_node.getHead()]))
        self._S_nodes_by_head_tuple[s_node.getHead()].remove(s_node)
        for tidx in range(s_node.getNumberTail()):
            self._S_nodes_by_tail_tuple[s_node.getTail(tidx)].remove(s_node)
        self._S_nodes.remove(s_node)
        self._modified = True


    def addSNode(self, s_node):
        # Returns False if found in BKB
        # NOTE -- Takes ownership of s_node and will deallocate when not needed
        if s_node in self._S_nodes:
            return (False)
        self._S_nodes.add(s_node)
        self._S_nodes_by_head_tuple[s_node.getHead()].add(s_node)
        for tidx in range(s_node.getNumberTail()):
            self._S_nodes_by_tail_tuple[s_node.getTail(tidx)].add(s_node)
        self._modified = True
        return (True)


    def getSrcComponents(self): # Returns a set of component indices
        if self._modified or len(self._src_components) == 0:
            for comp_name, cidx in self._components_index.items():
                component = self._components[cidx]
                if component.match('Source', contains=True) or component.match('source', contains=True):
                    self._src_components.add(cidx)
        return (copy.deepcopy(self._src_components))

    def getComponentsContains(self, substr): # Returns a set of component indices that contains substr
        indices = set()
        for comp_name, cidx in self._components_index.items():
            component = self._components[cidx]
            if component.match(substr, contains=True):
                indices.add(cidx)
        return (indices)

    def getName(self):
        return (copy.deepcopy(self._name))

    def getAllSNodes(self):
        return (copy.copy(self._S_nodes))

    def constructSNodesByHead(self):
        return (copy.copy(self._S_nodes_by_head_tuple))

    def constructSNodesByTail(self):
        return (copy.copy(self._S_nodes_by_tail_tuple))

    def serialize(self):
        return compress_pickle.dumps(self, compression="lz4")

    def calculate_mdl(self, data, feature_states):
        node_encoding_len = np.log2(len(self.getINodeNames()))
        feature_states_map = {(feature, state): idx for idx, (feature, state) in enumerate(feature_states)}
        mdl = 0
        store = None
        for snode in self.getAllSNodes():
            snode_mdl, store = snode.calculate_mdl(
                    self,
                    node_encoding_len,
                    data,
                    feature_states,
                    feature_states_map,
                    store,
                    )
            mdl += snode_mdl
        return mdl
    
    def calculate_mdl_ent(self, data, feature_states, num_inodes_override=None):
        if num_inodes_override is None:
            node_encoding_len = np.log2(len(self.getINodeNames()))
        else:
            node_encoding_len = np.log2(num_inodes_override)
        feature_states_map = {(feature, state): idx for idx, (feature, state) in enumerate(feature_states)}
        mdl = 0
        store = None
        for snode in self.getAllSNodes():
            snode_mdl, store = snode.calculate_mdl_ent(
                    self,
                    node_encoding_len,
                    data,
                    feature_states,
                    feature_states_map,
                    store,
                    )
            mdl += snode_mdl
        return mdl

    def is_mutex(self):
        snodes_by_head = self.constructSNodesByHead()
        for head, snodes in snodes_by_head.items():
            for snode1, snode2 in itertools.combinations(snodes, r=2):
                if not snode1.isMutex(snode2):
                    return False
        return True

    def to_str(self):
        bkb_str = ''
        for cid, comp in sorted(self._components.items()):
            for iid, inode in sorted(comp._states.items()):
                bkb_str += self.getComponentName(cid) + self.getComponentINodeName(cid, iid)
                s_nodes = self._S_nodes_by_head_tuple[(cid,iid)]
                s_node_ids = list()
                for s_node in s_nodes:
                    s_node_id = str(s_node.probability)
                    # head node information
                    s_node_id += self.getComponentName(s_node.head[0])
                    s_node_id += self.getComponentINodeName(s_node.head[0], s_node.head[1])
                    # tail nodes information
                    tail_node_ids = list()
                    for tail in s_node.tail:
                        tail_id = self.getComponentName(tail[0])
                        tail_id += self.getComponentINodeName(tail[0], tail[1])
                        tail_node_ids.append(tail_id)
                    # sort tail node ids and concat to s_node id
                    tail_node_ids.sort()
                    for tail_node_id in tail_node_ids:
                        s_node_id += tail_node_id
                    s_node_ids.append(s_node_id)
                # sort s_node_ids and concat to bkb_str
                s_node_ids.sort()
                for s_node_id in s_node_ids:
                    bkb_str += s_node_id              
        return bkb_str
                
                

# Private functions

    def _getComponent(self, component_index): # Returns the component or -1
        try:
            return (self._components[component_index])
        except KeyError:
            return (-1)

    def _getComponentINode(self, component_index, state_index): # Returns the I-node or -1
        try:
            return (self._components[component_index]._getState(state_index))
        except KeyError:
            return (-1)

    def _findComponent(self, component_name): # Private function
        try:
            return (self._components[self._components_index[component_name]])
        except KeyError:
            return (-1)

    def _addComponent(self, comp): # Private function
        comp_name = comp.getName()
        cidx = self.addComponent(comp_name)
        indices = comp.getAllStateIndices()
        for iidx in indices:
            state_name = comp.getStateName(iidx)
            self.addComponentState(cidx, state_name)
        del indices

    @staticmethod
    def join(bkbs):
        bkb = bayesianKnowledgeBase()
        for _bkb in bkbs:
            for _snode in _bkb._S_nodes:
                # Convert head
                _head_comp_idx, _head_state_idx = _snode.getHead()
                head_comp = _bkb.getComponentName(_head_comp_idx)
                head_state = _bkb.getComponentINodeName(_head_comp_idx, _head_state_idx)
                head_comp_idx = bkb.addComponent(head_comp)
                head_state_idx = bkb.addComponentState(head_comp_idx, head_state)
                # Convert tail
                tail = []
                for _tail_idx in range(_snode.getNumberTail()):
                    _tail_comp_idx, _tail_state_idx = _snode.getTail(_tail_idx)
                    tail_comp = _bkb.getComponentName(_tail_comp_idx)
                    tail_state = _bkb.getComponentINodeName(_tail_comp_idx, _tail_state_idx)
                    tail_comp_idx = bkb.addComponent(tail_comp)
                    tail_state_idx = bkb.addComponentState(tail_comp_idx, tail_state)
                    tail.append((tail_comp_idx, tail_state_idx))
                # Add snode
                snode = BKB_S_node(
                        head_comp_idx,
                        head_state_idx,
                        _snode.getProbability(),
                        tail,
                        )
                if snode not in bkb._S_nodes:
                    bkb.addSNode(snode)
        return bkb

    def __eq__(self, other):
        if isinstance(other, bayesianKnowledgeBase):
            #-- Construct S-nodes by head
            snodes_by_head_self = self.constructSNodesByHead()
            snodes_by_head_other = other.constructSNodesByHead()
            #print(snodes_by_head_self)
            #print(snodes_by_head_other)

            #-- If there aren't the same number of head I-nodes
            if len(snodes_by_head_self) != len(snodes_by_head_other):
                #print('Failed 1')
                return False

            #-- If I-node sets does not perfectly match.
            if len(set(self.getINodeNames()) ^ set(other.getINodeNames())) != 0:
                #print('Failed 2')
                return False

            #-- If I-nodes and number of snodes are the same, check all equivalance of all snodes.
            for head_self, snodes_self in snodes_by_head_self.items():
                head_self_comp_name = self.getComponentName(head_self[0])
                head_self_inode_name = self.getComponentINodeName(head_self[0], head_self[1])
                for head_other, snodes_other in snodes_by_head_other.items():
                    head_other_comp_name = other.getComponentName(head_other[0])
                    head_other_inode_name = other.getComponentINodeName(head_other[0], head_other[1])
                    if head_other_comp_name == head_self_comp_name:
                        if head_other_inode_name == head_self_inode_name:
                            if len(snodes_self) != len(snodes_other):
                                #print('Failed 3')
                                return False
                            found_inode_math = True
                            matched_snodes = []
                            for snode_self in snodes_self:
                                for snode_other in snodes_other:
                                    if self.compare_snodes(snode_self, snode_other, self, other):
                                        matched_snodes.append(True)
                            if len(snodes_self) == len(matched_snodes):
                                break
                            else:
                                #print('Failed 4')
                                return False
            return True
        #print('Failed 5')
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    @staticmethod
    def compare_snodes(snode1, snode2, bkb, other_bkb=None):
        #-- First just compare probabilties
        if snode1.getProbability() != snode2.getProbability():
            return False
        #-- Make human readable snodes
        hr_snode1 = HumanReadableSnode.make(snode1, bkb)
        if other_bkb is None:
            hr_snode2 = HumanReadableSnode.make(snode2, bkb)
        else:
            hr_snode2 = HumanReadableSnode.make(snode2, other_bkb)
        return hr_snode1 == hr_snode2
