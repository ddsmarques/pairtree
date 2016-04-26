#include "GreedyBBTree.h"

#include "CompareUtils.h"


std::shared_ptr<DecisionTreeNode> GreedyBBTree::createTree(DataSet& ds, std::shared_ptr<ConfigTree> c) {
  std::shared_ptr<ConfigGreedyBB> config = std::static_pointer_cast<ConfigGreedyBB>(c);
  std::vector<bool> availableAttrib(ds.getTotAttributes(), true);
  return createTreeRec(ds, config->height, availableAttrib);
}


std::shared_ptr<DecisionTreeNode> GreedyBBTree::createTreeRec(DataSet& ds, int64_t height, std::vector<bool> availableAttrib) {
  ErrorUtils::enforce(ds.getTotClasses() > 0, "Invalid data set.");

  if (height == 0) {
    auto best = ds.getBestClass();

    std::shared_ptr<DecisionTreeNode> leaf = std::make_shared<DecisionTreeNode>(DecisionTreeNode::NodeType::LEAF);
    leaf->setName("LEAF " + std::to_string(best.first));
    leaf->setLeafValue(best.first);

    return leaf;
  }
  auto state = initState(availableAttrib, height);
  std::shared_ptr<DecisionTreeNode> bestBB = nullptr;
  double bestScore;
  do {
    auto currBB = createBB(ds, state);
    if (bestBB == nullptr || CompareUtils::compare(bestScore, currBB.second) < 0) {
      bestBB = currBB.first;
      bestScore = currBB.second;
    }
  } while (nextState(ds, state));

  std::shared_ptr<DecisionTreeNode> root = bestBB;
  auto node = root;
  int64_t count = 0;
  DataSet currDS = ds;
  while (node != nullptr) {
    availableAttrib[node->getAttribCol()] = false;
    std::pair<std::shared_ptr<DecisionTreeNode>, int64_t> next = std::make_pair(nullptr, -1);
    for (auto&& child : node->children_) {
      if (child.second->isLeaf()) {
        child.second = createTreeRec(currDS.getSubDataSet(node->getAttribCol(), child.first),
                                      height - count - 1, availableAttrib);
      }
      else {
        next = std::make_pair(child.second, child.first);
      }
    }
    currDS = currDS.getSubDataSet(node->getAttribCol(), next.second);
    node = next.first;
    count++;
  }

  return root;
}


bool GreedyBBTree::nextState(DataSet& ds,
                             std::vector<std::pair<int64_t, int64_t>>& state) {
  int64_t totAttributes = ds.getTotAttributes();

  std::vector<bool> usedAttributes(totAttributes);
  for (int i = 0; i < state.size(); i++) {
    usedAttributes[state[i].first] = true;
  }

  int lastChanged = -1;
  for (int i = state.size() - 1; i >= 0; i--) {
    // Tries to change the value of the backbone for node 'i'.
    // Only does it for nodes above the leaf
    if (i < state.size() - 1 && state[i].second < ds.getAttributeSize(state[i].first) - 1) {
      state[i].second++;
      lastChanged = i;
      break;
    }
    // Tries to use the next available attribute for node 'i'
    int64_t nextAttrib = state[i].first + 1;
    while (nextAttrib < totAttributes && usedAttributes[nextAttrib]) {
      nextAttrib++;
    }
    if (nextAttrib < totAttributes) {
      usedAttributes[state[i].first] = false;
      usedAttributes[nextAttrib] = true;
      state[i].first = nextAttrib;
      state[i].second = 0;
      lastChanged = i;
      break;
    }
    usedAttributes[state[i].first] = false;
  }

  if (lastChanged == -1) {
    return false;
  }

  // Creates the attributes for the next nodes of this state
  for (int i = lastChanged + 1; i < state.size(); i++) {
    int64_t nextAttrib = 0;
    while (nextAttrib < totAttributes && usedAttributes[nextAttrib]) {
      nextAttrib++;
    }
    usedAttributes[nextAttrib] = true;
    state[i].first = nextAttrib;
    state[i].second = 0;
  }
  return true;
}


std::pair<std::shared_ptr<DecisionTreeNode>, double> GreedyBBTree::createBB(DataSet& ds,
                                                         const std::vector<std::pair<int64_t, int64_t>>& state) {
  std::shared_ptr<DecisionTreeNode> root = nullptr;
  double finalScore = 0;
  
  DataSet currDS = ds;
  std::shared_ptr<DecisionTreeNode> prev = nullptr;
  for (int i = 0; i < state.size(); i++) {
    std::shared_ptr<DecisionTreeNode> curr =
      std::make_shared<DecisionTreeNode>(DecisionTreeNode::NodeType::REGULAR_NOMINAL,
                                         state[i].first);
    for (int j = 0; j < currDS.getAttributeSize(state[i].first); j++) {
      if (j != state[i].second || i == state.size() - 1) {
        std::shared_ptr<DecisionTreeNode> leaf =
          std::make_shared<DecisionTreeNode>(DecisionTreeNode::NodeType::LEAF);
        std::pair<int64_t, double> bestClass = currDS.getSubDataSet(state[i].first, j).getBestClass();
        finalScore += bestClass.second;
        leaf->setLeafValue(bestClass.first);
        curr->addChild(leaf, { j });
      }
    }

    if (root == nullptr) {
      root = curr;
    }
    if (prev != nullptr) {
      prev->addChild(curr, {state[i-1].second});
    }
    prev = curr;
    currDS = currDS.getSubDataSet(state[i].first, state[i].second);
  }

  return std::make_pair(root, finalScore);
}


std::vector<std::pair<int64_t, int64_t>> GreedyBBTree::initState(std::vector<bool> availableAttrib,
                                                                 int64_t height) {
  std::vector<std::pair<int64_t, int64_t>> ans(height);

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < availableAttrib.size(); j++) {
      if (availableAttrib[j]) {
        availableAttrib[j] = false;
        ans[i] = std::make_pair(j, 0);
        break;
      }
    }
  }

  return ans;
}
