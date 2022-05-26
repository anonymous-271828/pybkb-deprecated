import sys
from math import floor
from math import log
from math import exp
import os
import csv
import copy
import parse
import time
import tqdm
import BKBReasoning
from docplex.mp.model import Model

DEBUG = True 

# This version constructs a MBLP to compute belief revision and updating.
# Constructs both BARON and CPLEX MBLPs
# Assumes the BKB is a valid construction

# Note, we rely on BKBReasoning.setup(bkb) for indexing

# This function is the primary function for belief updating and only employs
#   BARON -- replaces BKBReasoning._computeJoint(...)
def _computeJoint(bkb, target_inodes, target_components, descriptor, exe_fn, maxtime, license_fn, num_solns, external = False):
    global var_desc_to_var
    global var_to_var_desc
    global cplex_variables
    if not external:
        model = Model()
        optimizer = "CPLEX"
    else:
        model = None
        optimizer = "BARON"
    start_time = time.time()
    BKBReasoning.setup(bkb)

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

    isfeasible, objective_value, answers = solveBKB(bkb, fixed_target_components, target_inodes, descriptor, optimizer, exe_fn, maxtime, license_fn, num_solns, model)
    if not isfeasible:
        return ( [ set(), set(), set(), 0, time.time() - start_time ] )


    # Process answers
    _answers = {}
    for a_idx, answer in answers.items():
        inode_set = set()
        for varname, value in answer.items():
            if value != 1: # Skip since not set
                continue
            inode_name = var_to_var_desc[varname]
#            print (inode_name)
            e = parse.parse("I_<{}={}>", inode_name, case_sensitive=True)
            if e != None:
                try:
                    component_idx = int(e[0])
                    state_idx = int(e[1])
                    inode_set.add((component_idx, state_idx))
                except ValueError:
                    pass
        _answers[a_idx] = inode_set
#        if DEBUG:
#            print ("Answer #{} --".format(a_idx))
#            print ("\tProbability = {}".format(exp(objective_value[a_idx])))
#            print ("\t{}".format(inode_set))

    # Build a maximal set of inferences such that there are no supergraphs
    #   between any two inferences
    mutex = {}
    for a_idx, _answer in _answers.items():
        skip = False
        for idx, m in [ (k, v) for (k, v) in mutex.items() ]:
            if m.issubset(_answer):
                skip = True
                break
            if _answer.issubset(m):
                del mutex[idx]
        if not skip:
            mutex[a_idx] = _answer
#    if DEBUG:
#        print (mutex)

    
    contributions = {} 
    probabilities = {}
    completed_inferences = {}
    for a_idx, _answer in mutex.items():
        ans = frozenset(_answer & target_inodes_filter)
        prob = exp(objective_value[a_idx])
        try:
            probabilities[ans] += prob
        except KeyError:
            probabilities[ans] = prob
            completed_inferences[ans] = list()
            contributions[ans] = {}
        snodes = getAllCompatibleSNodes(bkb, _answer)
        completed_inferences[ans].append([ prob, _answer, snodes, [ c_idx for ( c_idx, _ ) in _answer ], set(), set() ])
        for s_idx in snodes:
            try:
                contributions[ans][s_idx] += prob
            except KeyError:
                contributions[ans][s_idx] = prob

    return ( [ probabilities, contributions, completed_inferences, len(mutex), time.time() - start_time ] )

# This function is the primary function for belief revision and can use either
#   CPLEX or BARON

def getAllCompatibleSNodes(bkb, inode_set):
    snodes = list()
    for _inode_set, s_idx in bkb.__BKBR_snodes_by_inodes.items():
        if _inode_set.issubset(inode_set):
            snodes.append(s_idx)
    return (snodes)

#
# The following defines the MBLP formulater and solver.

# Notation:
#   Component   -- "C_{}".format(component_idx)
#                   -- component_idx is the index of the cth component
#   I-Node      -- "I_<{}={}>".format(component_idx, state_idx)
#                   -- component_idx is the index of the cth component
#                   -- state_idx is the index of the vth state of cth component
#   S-Node      -- "S_<{}={}>_{}".format(component_idx, state_idx, snode_idx)
#                   -- This is an S-Node supporting (has head) of I-Node
#                       "I_<{}={}".format(component_idx, state_idx)
#                   -- snode_idx is the index of this S-node in __BKBR_snodes
#               -- "logP(S_<{}={}>_{}".format(component_idx, state_idx, snode_idx)
#                   -- This is the log probability of this S-Node
#               -- "tail_S_<{}={}>_{}".format(component_idx, state_idx, snode_idx)
#                   --- Is the set of I-Nodes in the tail of of this S-Node

#
# Dictionaries:
#
#   var_desc_to_var                 -- dictionary mapping from the descriptor to name (global)
#   var_to_var_desc                 -- dictionary mapping from name to descriptor (global)
#   cplex_variables                 -- dictionary mapping from variable name to a cplex variable
#   equation_desc_to_equation       -- dictionary mapping from the descrtiptor to name (global)
#   equation_to_equation_desc       -- dictionary mapping from name to descriptor (global)

var_desc_to_var = dict()
var_to_var_desc = dict()
cplex_variables = dict()
equation_desc_to_equation = dict()
equation_to_equation_desc = dict()

#
# Other globals
#
#   next_var_idx                    -- next available integer index for vars (global)
#   next_equation_idx               -- next available integer index for equations (global)
next_var_idx = 0
next_equation_idx = 0

#
# Variables: (bounds are establish at time of declaration)
#
#   "I_<{}={}>".format(component_idx, state_idx)
#       -- binary variable (0 = False, 1 = True)
#   "S_<{}={}>_{}".format(component_idx, state_idx, snode_idx)
#       -- binary variable

#
# Constraints: (unbound indices are assumed universal)
#
# C:
#   \sum{state_idx} "I_<{}={}>".format(component_idx, state_idx) <= 1
#       -- At most one I-Node per component can be selected
# I:
#   "I_<{}={}>".format(component_idx, state_idx)
#       = \sum_{snode_idx} "S_<{}={}>_{}".format(component_idx, state_idx, snode_idx)
#       -- An I-Node is selected iff at least one S-Node supporting
#           it is selected.
# S:
#   "S_<{}={}>_{}".format(component_idx, state_idx, snode_idx) * 
#       len("tail_S_<{}={}>_{}".format(component_idx, state_idx, snode_idx))
#       <= \sum_{(c'_idx, v'_idx)
#           \in "tail_S_<{}={}>_{}".format(component_idx, state_idx, snode_idx)}
#           "I_<{}={}>".format(c'_idx, v'_idx) 
#
# evidence:
#
#   if "C_{}".format(component_idx) is set as evidence, then add constraint:
# CE:
#       \sum{state_idx} "I_<{}={}>".format(component_idx, state_idx) >= 1
#           -- Exactly one I-Node per component can be selected
#
#   if "I_<{}={}>".format(component_idx, state_idx) is set as evidence, then
#       add constraint:
# IE:      
#      "I_<{}={}>".format(component_idx, state_idx) = 1
#

#   
# Objective function:
#       max \sum_{component_idx, state_idx, snode_idx}
#           "logP(S_<{}={}>_{})".format(component_idx, state_idx, snode_idx)
#           * "S_<{}={}>_{}".format(component_idx, state_idx, snode_idx)
#       -- Finds the highest probability inference graph satisfying the
#           given evidence. Without evidence, the empty graph is returned

#
# Generate variable strings

def genVariableName(desc, prefix):
    global next_var_idx
    try:
        return (var_desc_to_var[desc])
    except KeyError:
        name = "{}{}".format(prefix, next_var_idx)
        var_desc_to_var[desc] = name
        var_to_var_desc[name] = desc
        next_var_idx += 1
        return (name)

#   "I_<{}={}>".format(component_idx, state_idx)
#       -- binary variable (0 = False, 1 = True)
def genI(component_idx, state_idx):
    desc = "I_<{}={}>".format(component_idx, state_idx)
    return (genVariableName(desc, "I"))

#   "S_<{}={}>_{}".format(component_idx, state_idx, snode_idx)
#       -- binary variable
def genS(component_idx, state_idx, snode_idx):
    desc = "S_<{}={}>_{}".format(component_idx, state_idx, snode_idx)
    return (genVariableName(desc, "S"))

#
# Generate variable type -- Optimizer specific (BARON)

def typeI():
    return ("BINARY_VARIABLE")

def typeS():
    return ("BINARY_VARIABLE")

#
# Generate variable bounds based on variable strings -- Optimizer specific (BARON)

#def upperbounds(state_idx, feature_idx):
#    return ("{}: 1;".format(gens(state_idx, feature_idx)))

#
# Generate constraints based on variable strings -- Optimizer specific
#   -- only if model is not specified is True (BARON)
#   -- otherwise, generates CPLEX/DOCPLEX constraint expressions in model

def genEquationName(desc, prefix):
    global next_equation_idx
    try:
        return (equation_desc_to_equation[desc])
    except KeyError:
        name = "{}{}".format(prefix, next_equation_idx)
        next_equation_idx += 1
        equation_desc_to_equation[desc] = name
        equation_to_equation_desc[name] = desc
        return (name)

# C:
#   \sum{state_idx} "I_<{}={}>".format(component_idx, state_idx) <= 1
#       -- At most one I-Node per component can be selected
def constraintC_name(component_idx):
    desc = "_C_{}".format(component_idx)
    return (genEquationName(desc, "C_"))
    
def constraintC(component_idx, state_idx_list, model = None):
    if model == None:
        s = ""
        for idx, state_idx in enumerate(state_idx_list):
            s += "{}".format(genI(component_idx, state_idx))
            if idx < len(state_idx_list) - 1:
                s += " + "
        return ("{} <= 1;".format(s))
    else:
        model.add_constraint_(model.sum(cplex_variables[genI(component_idx, state_idx)] for state_idx in state_idx_list) <= 1, constraintC_name(component_idx))
        return (None)

# I:
#   "I_<{}={}>".format(component_idx, state_idx)
#       = \sum_{snode_idx} "S_<{}={}>_{}".format(component_idx, state_idx, snode_idx)
#       -- I-Node is selected iff at least one S-Node supporting
#           it is selected.
def constraintI_name(component_idx, state_idx):
    desc = "_I_<{}={}>".format(component_idx, state_idx)
    return (genEquationName(desc, "I_"))
    
def constraintI(component_idx, state_idx, snode_idx_list, model = None):
    if model == None:
        s = genI(component_idx, state_idx)
        for snode_idx in snode_idx_list:
            s += " - {}".format(genS(component_idx, state_idx, snode_idx))
        return ("{} == 0;".format(s))
    else:
        s = -1 * cplex_variables[genI(component_idx, state_idx)]
        s += model.sum(cplex_variables[genS(component_idx, state_idx, snode_idx)] for snode_idx in snode_idx_list) 
        model.add_constraint_(s == 0, constraintI_name(component_idx, state_idx))
        return (None)
    
# S:
#   "S_<{}={}>_{}".format(component_idx, state_idx, snode_idx) * 
#       len("tail_S_<{}={}>_{}".format(component_idx, state_idx, snode_idx))
#       <= \sum_{(c'_idx, v'_idx)
#           \in "tail_S_<{}={}>_{}".format(component_idx, state_idx, snode_idx)}
#           "I_<{}={}>".format(c'_idx, v'_idx) 
def constraintS_name(component_idx, state_idx, snode_idx):
    desc = "_S_<{}={}>_{}".format(component_idx, state_idx, snode_idx)
    return (genEquationName(desc, "S_"))
    
def constraintS(component_idx, state_idx, snode_idx, tail_inode_list, model = None):
    if model == None:
        s = "{} * {}".format(len(tail_inode_list), genS(component_idx, state_idx, snode_idx))
        for c_idx, v_idx in tail_inode_list:
            s += " - {}".format(genI(c_idx, v_idx))
        return ("{} <= 0;".format(s))
    else:
        s = -len(tail_inode_list) * cplex_variables[genS(component_idx, state_idx, snode_idx)]
        s += model.sum(cplex_variables[genI(c_idx, v_idx)] for (c_idx, v_idx) in tail_inode_list)
        model.add_constraint_(s >= 0, constraintS_name(component_idx, state_idx, snode_idx))
        return (None)
    
# evidence:
#
#   if "C_{}".format(component_idx) is set as evidence, then add constraint:
# CE:
#       \sum{state_idx} "I_<{}={}>".format(component_idx, state_idx) >= 1
#           -- Exactly one I-Node per component can be selected
#
#   if "I_<{}={}>".format(component_idx, state_idx) is set as evidence, then
#       add constraint:
def constraintCE_name(component_idx):
    desc = "_CE_{}".format(component_idx)
    return (genEquationName(desc, "CE_"))
    
def constraintCE(component_idx, state_idx_list, model = None):
    if model == None:
        s = ""
        for idx, state_idx in enumerate(state_idx_list):
            s += "{}".format(genI(component_idx, state_idx))
            if idx < len(state_idx_list) - 1:
                s += " + "
        return ("{} >= 1;".format(s))
    else:
        model.add_constraint_(model.sum(cplex_variables[genI(component_idx, state_idx)] for state_idx in state_idx_list) >= 1, constraintCE_name(component_idx))
        return (None)
    
# IE:      
#      "I_<{}={}>".format(component_idx, state_idx) = 1
def constraintIE_name(component_idx, state_idx):
    desc = "_IE_<{}={}>".format(component_idx, state_idx)
    return (genEquationName(desc, "IE_"))
    
def constraintIE(component_idx, state_idx, model = None):
    if model == None:
        return ("{} == 1;".format(genI(component_idx, state_idx)))
    else:
        model.add_constraint_(cplex_variables[genI(component_idx, state_idx)] == 1, constraintIE_name(component_idx, state_idx))
        return (None)
    
#
# Declare all variables -- specific to BARON
def addVariable(var, var_type, binary_variables, integer_variables, positive_variables, variables):
    if var_type == "BINARY_VARIABLE":
        if var in binary_variables:
#            if DEBUG:
#                print("{} already in binary variables!".format(var))
            return (False)
        else:
            binary_variables.add(var)
            return (True)
    if var_type == "INTEGER_VARIABLE":
        if var in integer_variables:
#            if DEBUG:
#                print("{} already in integer variables!".format(var))
            return (False)
        else:
            integer_variables.add(var)
            return (True)
    if var_type == "POSITIVE_VARIABLE":
        if var in positive_variables:
#            if DEBUG:
#                print("{} already in positive variables!".format(var))
            return (False)
        else:
            positive_variables.add(var)
            return (True)
    if var_type == "VARIABLE":
        if var in variables:
#            if DEBUG:
#                print("{} already in variables!".format(var))
            return (False)
        else:
            variables.add(var)
            return (True)
    sys.exit("{} type unknown for variable <{}>".format(var_type, var)) 
        
def createDeclarations(type, variables):
    vars = list(variables)
    vars.sort()
    row = "{} {}".format(type, vars[0])
    for idx in range(1, len(vars)):
        row += ",\n\t{}".format(vars[idx])
    row += ";"
    del vars
    return (row)
    
def declareVariables(bkb, model = None):
    BKBReasoning.setup(bkb)
    # Returns a tuple with the first element a list of output strings each a
    # row if model is not specified
    #   Otherwise returns None
    if model == None:
        rows = list()
        binary_variables = set()
        integer_variables = set()
        positive_variables = set()
        variables = set()
   
    component_indices = bkb.__BKBR_components
  
    #   "I_<{}={}>".format(component_idx, state_idx)
    #       -- binary variable (0 = False, 1 = True)
    if DEBUG:
        print ('...I variables...')
        var_ct = len(var_to_var_desc)
    for component_idx in component_indices:
        state_indices = [ state_idx for (_, state_idx) in bkb.__BKBR_inodes_in_component[component_idx] ]
        for state_idx in state_indices:
            if model == None:
                addVariable(genI(component_idx, state_idx), typeI(), binary_variables, integer_variables, positive_variables, variables)
            else:
                if typeI() == "BINARY_VARIABLE":
                    name = genI(component_idx, state_idx)
                    if not name in cplex_variables.keys():
                        cplex_variables[name] = model.binary_var(name)
                else:
                    print ("I variables only implemented as binary valued.")
                    sys.exit(-1)
    if DEBUG:
        print ('......{} declared.'.format(len(var_to_var_desc) - var_ct))

    #   "S_<{}={}>_{}".format(component_idx, state_idx, snode_idx)
    #       -- binary variable
    if DEBUG:
        print ('...S variables...')
        var_ct = len(var_to_var_desc)
    for component_idx in component_indices:
        state_indices = [ state_idx for (_, state_idx) in bkb.__BKBR_inodes_in_component[component_idx] ]
        for state_idx in state_indices:
            for snode_idx in bkb.__BKBR_snodes_by_head[( component_idx, state_idx )]:
                if model == None:
                    addVariable(genS(component_idx, state_idx, snode_idx), typeS(), binary_variables, integer_variables, positive_variables, variables)
                else:
                    if typeI() == "BINARY_VARIABLE":
                        name = genS(component_idx, state_idx, snode_idx)
                        if not name in cplex_variables.keys():
                            cplex_variables[name] = model.binary_var(name)
                    else:
                        print ("S variables only implemented as binary valued.")
                        sys.exit(-1)
    if DEBUG:
        print ('......{} declared.'.format(len(var_to_var_desc) - var_ct))

    # Generate declarations
   
    if model == None: 
        if len(binary_variables) > 0:
            rows.append(createDeclarations("BINARY_VARIABLES", binary_variables))
        if len(integer_variables) > 0:
            rows.append(createDeclarations("INTEGER_VARIABLES", integer_variables))
        if len(positive_variables) > 0:
            rows.append(createDeclarations("POSITIVE_VARIABLES", positive_variables))
        if len(variables) > 0:
            rows.append(createDeclarations("VARIABLES", variables))
        del binary_variables
        del integer_variables
        del positive_variables
        del variables

    if DEBUG:
        print ("Total variables declared = {}".format(len(var_to_var_desc)))
    if model == None:
        return (rows)
    else:
        return (None)
    
#
# Bound variables
def boundVariables(bkb, optimizer, model = None):
# If model is not specified, returns rows for file specification
#   otherwise returns None
    BKBReasoning.setup(bkb)
    # Returns a list of rows detailing the bounds
    if model == None:
        rows = list()
    if optimizer == "BARON":
        if model == None:
            return (list()) ########################################## None
        else:
            return (None)

        rows.append("LOWER_BOUNDS{")
    
#        if not simplify_masking:
#            for snode_idx in range(num_states):
#                rows.append(lowerboundQ(snode_idx))
                
    
        rows.append("}")
        rows.append(" ")
        
        rows.append("UPPER_BOUNDS{")
    
#        for snode_idx in range(num_states):
#            if not simplify_masking:
#                rows.append(upperboundQ(snode_idx, num_classes))
#            if not simplify_masking:
#                for f_idx1 in range(num_features):
#                    rows.append(upperbounds(snode_idx, f_idx1))
    
        rows.append("}")
    if optimizer == "CPLEX":

        if model == None:
            return (list()) ########################################## None
        else:
            return (None)

        rows.append("Bounds")
        # POSITIVE VARIABLES have a default of non-negative in CPLEX
        # Need to take care of "VARIABLE" if not yet bound using "-infinity"
#        for p_idx in range(num_posets - 1):
#            lb = lowerbounddelta(p_idx, p_idx + 1)
#            idx = lb.find(":")
#            n = lb[:idx]
#            lv = lb[idx+1:-1]
#            rows.append("{} <= {}".format(lv, n))
#
#        for t_idx, traj in enumerate(trajectories):
#            rows.append("-infinity <= {}".format(genLER(t_idx)))
#
#        for p_idx in range(num_posets):
#            rows.append("-infinity <= {}".format(genLB(p_idx)))
#            rows.append("-infinity <= {}".format(genUB(p_idx)))
        rows.append("End")
    if model == None:
        return (rows)
    else:
        return (None)
    
#
# Build constraints
def generateConstraints(bkb, component_evidence_list, inode_evidence_list, optimizer, model = None):
    # component_evidence_list is a list of component indices
    # inode_evidence_list is a list of tuples (component_idx, state_idx)

    BKBReasoning.setup(bkb)
    # Returns a list of rows detailing the constraint declarations and constraints -- specific to (BARON)
    equation_names = list()
    equation_names_set = set()
    equations = list()

    component_indices = bkb.__BKBR_components

    # C:
    #   \sum{state_idx} "I_<{}={}>".format(component_idx, state_idx) <= 1
    #       -- At most one I-Node per component can be selected
    if DEBUG:
        print ("...C constraints...")
    num_constraints = 0
    for component_idx in component_indices:
        if len(bkb.__BKBR_inodes_in_component[component_idx]) <= 1:
            continue
        eqn = constraintC_name(component_idx)
        if not eqn in equation_names_set: # skip
            equation_names_set.add(eqn)
            equation_names.append(eqn)
            state_indices = [ state_idx for (_, state_idx) in bkb.__BKBR_inodes_in_component[component_idx] ]
            if model == None:
                equations.append(constraintC(component_idx, state_indices))
            else:
                constraintC(component_idx, state_indices, model)
            num_constraints += 1
    if DEBUG:
        print ("......{} created.".format(num_constraints))

    # I:
    #   "I_<{}={}>".format(component_idx, state_idx)
    #       = \sum_{snode_idx} "S_<{}={}>_{}".format(component_idx, state_idx, snode_idx)
    #       -- I-Node is selected iff at least one S-Node supporting
    #           it is selected.
    if DEBUG:
        print ("...I constraints...")
    num_constraints = 0
    for component_idx in component_indices:
        state_indices = [ state_idx for (_, state_idx) in bkb.__BKBR_inodes_in_component[component_idx] ]
        for state_idx in state_indices:
            snode_indices = bkb.__BKBR_snodes_by_head[( component_idx, state_idx)]
            eqn = constraintI_name(component_idx, state_idx)
            if not eqn in equation_names_set: # skip
                equation_names_set.add(eqn)
                equation_names.append(eqn)
                if model == None:
                    equations.append(constraintI(component_idx, state_idx, snode_indices))
                else:
                    constraintI(component_idx, state_idx, snode_indices, model)
                num_constraints += 1
    if DEBUG:
        print ("......{} created.".format(num_constraints))

    # S:
    #   "S_<{}={}>_{}".format(component_idx, state_idx, snode_idx) * 
    #       len("tail_S_<{}={}>_{}".format(component_idx, state_idx, snode_idx))
    #       <= \sum_{(c'_idx, v'_idx)
    #           \in "tail_S_<{}={}>_{}".format(component_idx, state_idx, snode_idx)}
    #           "I_<{}={}>".format(c'_idx, v'_idx) 
    if DEBUG:
        print ("...S constraints...")
    num_constraints = 0
    for component_idx in component_indices:
        state_indices = [ state_idx for (_, state_idx) in bkb.__BKBR_inodes_in_component[component_idx] ]
        for state_idx in state_indices:
            snode_indices = bkb.__BKBR_snodes_by_head[( component_idx, state_idx)]
            for snode_idx in snode_indices:
                tail_inode_list = bkb.__BKBR_snodes[snode_idx].tail
                if len(tail_inode_list) == 0:
                    continue
                eqn = constraintS_name(component_idx, state_idx, snode_idx)
                if not eqn in equation_names_set: # skip
                    equation_names_set.add(eqn)
                    equation_names.append(eqn)
                    if model == None:
                        equations.append(constraintS(component_idx, state_idx, snode_idx, tail_inode_list))
                    else:
                        constraintS(component_idx, state_idx, snode_idx, tail_inode_list, model)
                    num_constraints += 1
    if DEBUG:
        print ("......{} created.".format(num_constraints))

    #
    # evidence:
    #
    #   if "C_{}".format(component_idx) is set as evidence, then add constraint:
    # CE:
    #       \sum{state_idx} "I_<{}={}>".format(component_idx, state_idx) >= 1
    #           -- Exactly one I-Node per component can be selected
    if DEBUG:
        print ("...CE constraints...")
    num_constraints = 0
    for component_idx in component_evidence_list:
        if len(bkb.__BKBR_inodes_in_component[component_idx]) == 0:
            continue
        eqn = constraintCE_name(component_idx)
        if not eqn in equation_names_set: # skip
            equation_names_set.add(eqn)
            equation_names.append(eqn)
            state_indices = [ state_idx for (_, state_idx) in bkb.__BKBR_inodes_in_component[component_idx] ]
            if model == None:
                equations.append(constraintCE(component_idx, state_indices))
            else:
                constraintCE(component_idx, state_indices, model)
            num_constraints += 1
    if DEBUG:
        print ("......{} created.".format(num_constraints))

    #
    #   if "I_<{}={}>".format(component_idx, state_idx) is set as evidence, then
    #       add constraint:
    # IE:      
    #      "I_<{}={}>".format(component_idx, state_idx) = 1
    #
    if DEBUG:
        print ("...IE constraints...")
    num_constraints = 0
    for (component_idx, state_idx) in inode_evidence_list:
        eqn = constraintIE_name(component_idx, state_idx)
        if not eqn in equation_names_set: # skip
            equation_names_set.add(eqn)
            equation_names.append(eqn)
            if model == None:
                equations.append(constraintIE(component_idx, state_idx))
            else:
                constraintIE(component_idx, state_idx, model)
            num_constraints += 1
    if DEBUG:
        print ("......{} created.".format(num_constraints))

    if DEBUG:
        print ("Total constraints created = {}".format(len(equation_names)))

    if model == None:
        # Form into optimzer format
               
        rows = list()
        if optimizer == "BARON":     
            if len(equation_names) == 0:
                return (rows)
            row = "EQUATIONS {}".format(equation_names[0])
            for e_idx in range(1, len(equation_names)):
                row += ","
                rows.append(row)
                row = "\t{}".format(equation_names[e_idx])
            row += ";"
            rows.append(row)
    
        for e_idx in range(len(equations)):
            row = "{}:\t{}".format(equation_names[e_idx], equations[e_idx])
            rows.append(row)
        del equation_names
        del equations
        del equation_names_set
        return (rows)  
    else:
        return (None)

def genObjective(bkb, model = None):
    BKBReasoning.setup(bkb)
    # max \sum_{component_idx, state_idx, snode_idx}
    #   "logP(S_<{}={}>_{})".format(component_idx, state_idx, snode_idx)
    #   * "S_<{}={}>_{}".format(component_idx, state_idx, snode_idx)
    if model == None:
        rows = list()
        rows.append("OBJ: maximize")
        first = True
        component_indices = bkb.__BKBR_components
        for component_idx in component_indices:
            state_indices = [ state_idx for (_, state_idx) in bkb.__BKBR_inodes_in_component[component_idx] ]
            for state_idx in state_indices:
                for snode_idx in bkb.__BKBR_snodes_by_head[( component_idx, state_idx )]:
                    prob = bkb.__BKBR_snodes[snode_idx].probability
                    if prob == -1 or prob == 0: # skip
                        continue
                    
                    if first:
                        s = "{} * {}".format(log(prob), genS(component_idx, state_idx, snode_idx))
                        first = False
                    else:
                        if -log(prob) > 0:
                            s += " - {} * {}".format(-log(prob), genS(component_idx, state_idx, snode_idx))
      
        rows.append("\t{};".format(s))
        return (rows)
    else:
        first = True
        component_indices = bkb.__BKBR_components
        for component_idx in component_indices:
            state_indices = [ state_idx for (_, state_idx) in bkb.__BKBR_inodes_in_component[component_idx] ]
            for state_idx in state_indices:
                for snode_idx in bkb.__BKBR_snodes_by_head[( component_idx, state_idx )]:
                    prob = bkb.__BKBR_snodes[snode_idx].probability
                    if prob == -1 or prob == 0: # skip
                        continue
                    
                    if first:
                        s = log(prob) * cplex_variables[genS(component_idx, state_idx, snode_idx)]
                        first = False
                    else:
                        if -log(prob) > 0:
                            s -= -log(prob) * cplex_variables[genS(component_idx, state_idx, snode_idx)]
      
        model.maximize(s)
        return (None)
  
# Returns the a list of rows for constructing an optimization file 
def constructOptimization(bkb, component_evidence_list, inode_evidence_list, optimizer, maxtime, license_fn, num_solns, model = None):
    global var_desc_to_var
    global var_to_var_desc
    global cplex_variables
    global equation_desc_to_equation
    global equation_to_equation_desc
    global next_var_idx
    global next_equation_idx
    var_desc_to_var = dict()
    var_to_var_desc = dict()
    cplex_variables = dict()
    equation_desc_to_equation = dict()
    equation_to_equation_desc = dict()
    next_var_idx = 0
    next_equation_idx = 0


    start_time = time.time()
    if DEBUG:
        print ("Building BKB optimization problem...")

    if model == None:
        if optimizer == "BARON":
            rows = [ "OPTIONS {", "MaxTime:{};".format(maxtime), "LicName: \"{}\";".format(license_fn), "nlpsol: 10;", "numsol: {};".format(num_solns), "}", " " ]
        
            if DEBUG:
                print ("Declaring variables...")
            new_rows = declareVariables(bkb)
            rows.extend(new_rows)
            rows.append(" ")
            if DEBUG:
                print ("\t...elapsed time = {}".format(time.time() - start_time))
                start_time = time.time()
                print ("Constructing variable bounds...")
            rows.extend(boundVariables(bkb, optimizer))
            rows.append(" ")
            if DEBUG:
                print ("\t...elapsed time = {}".format(time.time() - start_time))
                start_time = time.time()
                print ("Generating constraints...")
            rows.extend(generateConstraints(bkb, component_evidence_list, inode_evidence_list, optimizer))
            rows.append(" ")
            rows.extend(genObjective(bkb))
    
        if optimizer == "CPLEX":
            rows = list()
            if DEBUG:
                start_time = time.time()
                print ("Declaring variables...")
            var_rows = declareVariables(bkb)
            # save for integer and binary variables below
            if DEBUG:
                print ("\t...elapsed time = {}".format(time.time() - start_time))
            newrows = genObjective(bkb)
            for row in newrows:
                row = row.replace("OBJ: ", "")
                row = row.replace("*", "")
                row = row.replace(";", "")
                rows.append(row)
            if DEBUG:
                start_time = time.time()
                print ("Generating constraints...")
            newrows = generateConstraints(bkb, component_evidence_list, inode_evidence_list, optimizer)
            if len(newrows) > 0:
                rows.append("Subject To")
            for row in newrows:
                if row == "":
                    continue
                row = row.replace("==", "=")
                row = row.replace(";", "")
                row = row.replace("*", "")
                rows.append(row)
            if DEBUG:
                print ("\t...elapsed time = {}".format(time.time() - start_time))
                start_time = time.time()
                print ("Constructing variable bounds...")
            newrows = boundVariables(bkb, optimizer)
            rows.extend(newrows)
            rows.append("END")
    
            newrows = list() 
            for row in var_rows:
                if row[0:17] == "INTEGER_VARIABLES":
                    row = row[18:]
                    row = row.replace("\n", "")
                    row = row.replace(";", "")
                    row = row.replace("\t", "")
                    vs = row.split(',')
                    newrows.append("GENERAL")
                    for v in vs:
                        newrows.append("\t{}".format(v))
                    rows.extend(newrows)
                if row[0:16] == "BINARY_VARIABLES":
                    row = row[17:]
                    row = row.replace("\n", "")
                    row = row.replace(";", "")
                    row = row.replace("\t", "")
                    vs = row.split(',')
                    newrows.append("BINARY")
                    for v in vs:
                        newrows.append("\t{}".format(v))
                    rows.extend(newrows)
    else:
        declareVariables(bkb, model)
        genObjective(bkb, model)
        generateConstraints(bkb, component_evidence_list, inode_evidence_list, optimizer, model)
        boundVariables(bkb, optimizer, model)

    if DEBUG:
        print ("\t...elapsed time = {}".format(time.time() - start_time))
    if model == None:
        return ( rows, var_to_var_desc, equation_to_equation_desc )
    else:
        return ( None, var_to_var_desc, equation_to_equation_desc )

def solveBKB(bkb, component_evidence_list, inode_evidence_list, descriptor, optimizer, exe_fn, maxtime, license_fn, num_solns, model = None):

    master_start_time = time.time()

    # Convert to absolute paths
    exe_fn = os.path.abspath(exe_fn)
    if license_fn != None:
        license_fn = os.path.abspath(license_fn)

    if model == None:
        # Make a results directory
        results_dir = "{}-{}".format(descriptor, optimizer)
        print ("Creating directory for results {}...".format(results_dir))
        if os.path.exists(results_dir):
            if not os.path.isdir(results_dir):
                sys.exit("solveBKB(...) -- {} is an existing file, cannot create directory.".format(results_dir))
            print ("Using existing directory {} -- will overwrite files.".format(results_dir))
        else:
            os.mkdir(results_dir)
        cwd = os.getcwd()
        os.chdir(results_dir)

    ( rows, var_to_var_desc, equation_to_equation_desc ) = constructOptimization(bkb, component_evidence_list, inode_evidence_list, optimizer, maxtime, license_fn, num_solns, model)

    if model == None:
        opt_file = "{}.lp".format(optimizer)
    
        with open (opt_file, "w") as f:
            f.write("\n".join(rows))
            f.close()
    
        # Run BARON to optimize and extracct answers
        if optimizer == "BARON":
            os.system("{} {}".format(exe_fn, opt_file))
            isfeasible, objective_value, answers = extractAnswersBARON(var_to_var_desc)
    
        # Run CPLEX to optimize and extracct answers
        if optimizer == "CPLEX":
            os.system("rm {}.sol".format(optimizer))
            os.system("echo \"read {}\" > {}.cplex".format(opt_file, optimizer))
    #        os.system("echo \"set preprocessing presolve no\" >> {}-{}.cplex".format(constructDescriptor, optimizer))
            os.system("echo \"optimize\" >> {}.cplex".format(optimizer))
            os.system("echo \"write {}.sol\" >> {}.cplex".format(optimizer, optimizer))
            os.system("echo \"quit\" >> {}.cplex".format(optimizer))
            os.system("{} -f {}.cplex".format(exe_fn, optimizer))
    #        isfeasible, objective_value, answers = extractAnswersCPLEX("{}-{}.sol".format(constructDescriptor, optimizer), "{}_soln-{}.csv.gz".format(constructDescriptor, optimizer), var_to_var_desc)
    
        os.chdir(cwd)
    else:
        global cplex_variables
        objective_value = dict()
        answers = dict()
        for count in tqdm.tqdm(range(num_solns), desc="Alt Solutions"):
            solution = model.solve()
            if solution == None:
                break
#            solution.print_mst()
            if count == 0:
                isfeasible = solution.is_feasible_solution() == True
            if solution.is_feasible_solution() != True:
                break
            inodes_set = set()
            objective_value[count] = solution.get_objective_value()
            answers[count] = dict()
            for var_name, cplex_var in cplex_variables.items():
                value = solution.get_value(cplex_var)
                answers[count][var_name] = value
                if var_name[0] == 'I' and value == 1:
                    inodes_set.add(cplex_var)
            if len(inodes_set) > 0:
                model.add_constraint_(model.sum(inode for inode in inodes_set) <= (len(inodes_set) - 1), "_CUT_{}".format(count))

    return (isfeasible, objective_value, answers)

# Returns the answers found by BARON as: ( feasible, objective_value, answers)
#   feasible is a boolean indicating if the problem was feasible
#   objective_value[x] is a dictionary of objective value for answer x
#       x == 0 is the best solution
#   answers[x] is a dictionary of variable values for answer x
#   Note - the remaining answers are unordered.
def extractAnswersBARON(var_to_var_desc):
    print ("Extracting answers --")
    answers = dict()
    objective_value = dict()
    best_soln_idx = 0
    answers[best_soln_idx] = dict()
    with open("res.lst", "r") as results_f:

        # First get the best solution -- this also gets us the variable
        #   indices corresponding to "Variable No." in res.lst to extract
        #   the alternative solutions
        idx2var = dict()
        flag = True
        while flag:
            line = results_f.readline()
            if line == "":
                break
            if line == "The best solution found is:\n":
                flag = False
        if flag:
            print ('No solution found!')
            return (False, None, None)
        results_f.readline()
        results_f.readline()
        var_idx = 0
        while True:
            line = results_f.readline()
            if line == "\n":
                break
#            print (line, end='')

            info = line.split()
            if len(info) != 4:
                continue
            try:
                print ("Variable {} already assigned to value {}!".format(info[0], answers[info[0]]))
                return (False, None, None)
            except KeyError:
                answers[best_soln_idx][info[0]] = float(info[2])
                idx2var[var_idx] = info[0]
                var_idx += 1
#                print ("Varaible {} = {}".format(info[0], answers[info[0]]))
        line = results_f.readline()
        try:
            pos = line.index(':')
        except ValueError:
            print ("Unable to find objective value!")
            return (None)
        if line[:pos] != "The above solution has an objective value of":
            print ("Unable to find objective value!")
            return (False, None, None)
        objective_value[best_soln_idx] = float(line[pos+1:])

        # Now load in the rest of the answers
        total_num_vars = var_idx
        soln_idx = 1
        results_f.seek(0)
        found = False
        while not found:
            line = results_f.readline()
            if line == '':
                break
            if "*** Normal completion ***" in line:
                found = True
                break
        if found:
            # Extract the alternative answers
            while True:
                found = False
                while not found:
                    line = results_f.readline()
                    if line == '':
                        break
                    if "Objective value is:" in line:
                        info = line.split()
                        objective_value[soln_idx] = float(info[4])
                        results_f.readline()
                        results_f.readline()
                        success = True
                        answers[soln_idx] = dict()
                        for var_idx in range(total_num_vars):
                            line = results_f.readline()
                            if line == '\n':
                                print ("Premature EOF reached.")
                                success = False
                                break
                            info = line.split()
                            assert var_idx == int(info[1]) - 1
                            answers[soln_idx][idx2var[var_idx]] = float(info[2])
                        if not success:
                            break
                        found = True
                        soln_idx += 1
                if not found:
                    break

#    outputs = list()
#    for item, value in answers.items():
#        outputs.append([ var_to_var_desc[item], value ])
#    outputs.sort()
#    with gzip.open(filename, "wt", newline='') as csv_file:
#        writer = csv.writer(csv_file)
#        writer.writerows(outputs)
#        csv_file.close()
#    del outputs
    return (True, objective_value, answers)

