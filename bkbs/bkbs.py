from pybkb.common.bayesianKnowledgeBase import bayesianKnowledgeBase as BKB
from pybkb.common.bayesianKnowledgeBase import BKB_S_node

def tuningPaperExampleBKB():
    bkb = BKB()

    # Build component A
    comp_a = bkb.addComponent('A')
    state_a_1 = bkb.addComponentState(comp_a, 'a1')
    state_a_2 = bkb.addComponentState(comp_a, 'a2')
    # Build component B
    comp_b = bkb.addComponent('B')
    state_b_1 = bkb.addComponentState(comp_b, 'b1')
    state_b_2 = bkb.addComponentState(comp_b, 'b2')
    state_b_3 = bkb.addComponentState(comp_b, 'b3')
    # Build component C
    comp_c = bkb.addComponent('C')
    state_c_1 = bkb.addComponentState(comp_c, 'c1')
    state_c_2 = bkb.addComponentState(comp_c, 'c2')
    state_c_3 = bkb.addComponentState(comp_c, 'c3')
    # Build component D
    comp_d = bkb.addComponent('D')
    state_d_1 = bkb.addComponentState(comp_d, 'd1')
    state_d_2 = bkb.addComponentState(comp_d, 'd2')
    state_d_3 = bkb.addComponentState(comp_d, 'd3')

    # Build S-nodes
    q_1 = BKB_S_node(comp_a, state_a_1, .3)
    q_2 = BKB_S_node(comp_a, state_a_2, .7)
    q_3 = BKB_S_node(comp_c, state_c_1, .2)
    q_4 = BKB_S_node(comp_c, state_c_3, .4)
    q_5 = BKB_S_node(comp_b, state_b_1, .1, [(comp_a, state_a_1)])
    q_6 = BKB_S_node(comp_b, state_b_2, .8, [(comp_a, state_a_1)])
    q_7 = BKB_S_node(comp_b, state_b_2, .5, [(comp_a, state_a_2), (comp_c, state_c_1)])
    q_8 = BKB_S_node(comp_b, state_b_2, .7, [(comp_a, state_a_2), (comp_c, state_c_3)])
    q_9 = BKB_S_node(comp_d, state_d_1, .4, [(comp_b, state_b_2)])
    q_10 = BKB_S_node(comp_c, state_c_2, .4, [(comp_d, state_d_1)])
    q_11 = BKB_S_node(comp_d, state_d_2, .6, [(comp_b, state_b_2)])
    q_12 = BKB_S_node(comp_b, state_b_3, .1, [(comp_a, state_a_1)])

    bkb.addSNode(q_1)
    bkb.addSNode(q_2)
    bkb.addSNode(q_3)
    bkb.addSNode(q_4)
    bkb.addSNode(q_5)
    bkb.addSNode(q_6)
    bkb.addSNode(q_7)
    bkb.addSNode(q_8)
    bkb.addSNode(q_9)
    bkb.addSNode(q_10)
    bkb.addSNode(q_11)
    bkb.addSNode(q_12)

    return bkb
