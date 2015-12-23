#include "PineTree.h"
#include "Converter.h"

#include <algorithm>
#include <set>
#include <utility>
#include <string>

PineTree::PineTree() : model_(env_) {}

std::shared_ptr<DecisionTreeNode> PineTree::createTree(DataSet& ds, std::shared_ptr<ConfigTree> c) {
  std::shared_ptr<ConfigPine> config = std::static_pointer_cast<ConfigPine>(c);
  if (config->height == 0) {
    auto best = ds.getBestClass();
    ErrorUtils::enforce(best.first >= 0, "Invalid class index");

    std::shared_ptr<DecisionTreeNode> leaf = std::make_shared<DecisionTreeNode>(DecisionTreeNode::NodeType::LEAF);
    leaf->setName("LEAF " + std::to_string(best.first));
    leaf->setLeafValue(best.first);

    return leaf;
  }

  std::shared_ptr<DecisionTreeNode> root = nullptr;
  if (config->type == ConfigPine::SolverType::CONTINUOUS_AFTER_ROOT) {
    root = createBackBone(ds, config->height, ConfigPine::SolverType::CONTINUOUS);
    
    int64_t rootBB = getBackBoneValue(ds, root->getAttribCol());
    for (auto&& child : root->children_) {
      if (child.first == rootBB) {
        child.second = createBackBone(ds.getSubDataSet(root->getAttribCol(), child.first), config->height - 1, ConfigPine::SolverType::CONTINUOUS);
        break;
      }
    }
  } else {
    root = createBackBone(ds, config->height, config->type);
  }
  
  auto node = root;
  int64_t count = 0;
  DataSet currDS = ds;
  while (node != nullptr && !node->isLeaf()) {
    auto next = getNextNodeBB(node);

    for (auto&& child : node->children_) {
      if (child.second->isLeaf()) {
        std::shared_ptr<ConfigPine> newConfig = std::make_shared<ConfigPine>(*config);
        if (newConfig->type == ConfigPine::SolverType::CONTINUOUS_AFTER_ROOT) {
          newConfig->type = ConfigPine::SolverType::CONTINUOUS;
        }
        newConfig->height = config->height - count - 1;
        child.second = createTree(currDS.getSubDataSet(node->getAttribCol(), child.first), newConfig);
      }
    }

    currDS = currDS.getSubDataSet(node->getAttribCol(), next.first);
    node = next.second;
    count++;
  }

  return root;
}

std::shared_ptr<DecisionTreeNode> PineTree::createBackBone(DataSet& ds,
                                                            int64_t bbSize,
                                                            ConfigPine::SolverType type) {
  varType_ = type == ConfigPine::SolverType::INTEGER ? GRB_INTEGER : GRB_CONTINUOUS;
  totClasses_ = ds.getTotClasses();
  totAttributes_ = ds.getTotAttributes();

  // Calculate the maximum number of values an attribute can take
  maxAttribSize_ = 0;
  for (int i = 0; i < totAttributes_; i++) {
    maxAttribSize_ = std::max(maxAttribSize_, ds.getAttributeSize(i));
  }

  // Gets the backbone value for each attribute.
  // This is the value with the highest frequency
  bbValue_.resize(totAttributes_ - 1);
  for (int64_t i = 0; i < totAttributes_ - 1; i++) {
    bbValue_[i] = getBackBoneValue(ds, i);
  }

  createVariables(ds, bbSize);
  model_.update();
  createObjFunction(ds, bbSize);
  createConstraints(ds, bbSize);
  model_.optimize();

  defineNodeLeaves(ds, bbSize);
  return mountBackbone(ds, bbSize);
}

void PineTree::createVariables(DataSet& ds, int64_t bbSize) {
  // Create all X_i_j variables
  // X[i][j] = 1 if attribute i is associated to node j. 0 otherwise.
  initMultidimension(X_, { totAttributes_ - 1, bbSize });
  for (int64_t i = 0; i < totAttributes_ - 1; i++) {
    for (int64_t j = 0; j < bbSize; j++) {
      X_[i][j] = model_.addVar(0, 1, 0, varType_, "X_" + std::to_string(i) + "_" + std::to_string(j));
    }
  }

  // Create all Z_e_j variables
  // Z[e][j] = 1 if sample goes to one of node j leafs. 0 otherwise.
  initMultidimension(Z_, { (int64_t)ds.samples_.size(), bbSize });
  for (int64_t e = 0; e < ds.samples_.size(); e++) {
    for (int64_t j = 0; j < bbSize; j++) {
      Z_[e][j] = model_.addVar(0, 1, 0, varType_, "Z_" + std::to_string(e) + "_" + std::to_string(j));
    }
  }

  // Create all W_c_j_h variables
  // W[c][j][h] = 1 if the h-th box from node j is in class c. 0 otherwise.
  initMultidimension(W_, { totClasses_, bbSize, maxAttribSize_ });
  for (int64_t c = 0; c < totClasses_; c++) {
    for (int64_t j = 0; j < bbSize; j++) {
      for (int64_t h = 0; h < maxAttribSize_; h++) {
        W_[c][j][h] = model_.addVar(0, 1, 0, varType_, "W_"
          + std::to_string(c) + "_" + std::to_string(j)
          + "_" + std::to_string(h));
      }
    }
  }

  // Create all V_e_c_j_h variables
  // V[e][c][j][h] = 1 if sample falls to the h-th box of node j, and this box has class c. 0 otherwise.
  initMultidimension(V_, { (int64_t)ds.samples_.size(), totClasses_, bbSize, maxAttribSize_ });
  for (int64_t e = 0; e < ds.samples_.size(); e++) {
    for (int64_t c = 0; c < totClasses_; c++) {
      for (int64_t j = 0; j < bbSize; j++) {
        for (int64_t h = 0; h < maxAttribSize_; h++) {
          V_[e][c][j][h] = model_.addVar(0, 1, 0, varType_, "V_"
            + std::to_string(e) + "_" + std::to_string(c)
            + "_" + std::to_string(j) + "_" + std::to_string(h));
        }
      }
    }
  }

  // Create all L_e_c variables
  // L[e][c] = 1 if last node puts sample e to the backbone classified as c
  initMultidimension(L_, { (int64_t)ds.samples_.size(), totClasses_ });
  for (int64_t e = 0; e < ds.samples_.size(); e++) {
    for (int64_t c = 0; c < totClasses_; c++) {
      L_[e][c] = model_.addVar(0, 1, 0, varType_, "L_" + std::to_string(e) + "_"
        + std::to_string(c));
    }
  }

  // Create all Y_c variables
  // Y[c] = 1 if the samples put to the backbone by the last node are classified as 'c'
  initMultidimension(Y_, { totClasses_ });
  for (int64_t c = 0; c < totClasses_; c++) {
    Y_[c] = model_.addVar(0, 1, 0, varType_, "Y_" + std::to_string(c));
  }
}

void PineTree::createObjFunction(DataSet& ds, int64_t bbSize) {
  // Create objective function
  // Contribution of all samples except the ones the last node put on the backbone
  GRBLinExpr objFun;
  for (int64_t j = 0; j < bbSize; j++) {
    for (int64_t h = 0; h < maxAttribSize_; h++) {
      for (int64_t c = 0; c < totClasses_; c++) {
        int64_t e = 0;
        for (const auto& s : ds.samples_) {
          objFun += s->benefit_[c] * V_[e][c][j][h];
          e++;
        }
      }
    }
  }
  // Contribution from all samples put on the backbone by the last node
  for (int64_t c = 0; c < totClasses_; c++) {
    int64_t e = 0;
    for (const auto& s : ds.samples_) {
      objFun += s->benefit_[c] * L_[e][c];
      e++;
    }
  }
  model_.setObjective(objFun, GRB_MAXIMIZE);
}

void PineTree::createConstraints(DataSet& ds, int64_t bbSize) {
  // Constraint X 1
  // Exactly one attribute per node
  for (int64_t j = 0; j < bbSize; j++) {
    GRBLinExpr expr;
    for (int64_t i = 0; i < totAttributes_ - 1; i++) {
      expr += X_[i][j];
    }
    model_.addConstr(expr, GRB_EQUAL, 1, "c_X1_" + std::to_string(j));
  }

  // Constraint X 2
  // At most one node can be matched to an attribute
  for (int64_t i = 0; i < totAttributes_ - 1; i++) {
    GRBLinExpr expr;
    for (int64_t j = 0; j < bbSize; j++) {
      expr += X_[i][j];
    }
    model_.addConstr(expr, GRB_LESS_EQUAL, 1, "c_X2_" + std::to_string(i));
  }

  // Constraint Z 1
  // Guarantees sample e will be put out of the backbone by the attribute chosen in node j
  int64_t e = 0;
  for (const auto& s : ds.samples_) {
    // j = 0 is treated seperately
    GRBLinExpr expr;
    expr += Z_[e][0];
    for (int64_t i = 0; i < totAttributes_ - 1; i++) {
      if (s->inxValue_[i] != bbValue_[i]) {
        expr += -1 * X_[i][0];
      }
    }
    model_.addConstr(expr, GRB_EQUAL, 0, "c_Z1_" + std::to_string(e) + "_" + "0");

    // j > 0
    for (int64_t j = 1; j < bbSize; j++) {
      expr.clear();
      expr += Z_[e][j];
      for (int64_t i = 0; i < totAttributes_ - 1; i++) {
        if (s->inxValue_[i] != bbValue_[i]) {
          expr += -1 * X_[i][j];
        }
      }
      model_.addConstr(expr, GRB_LESS_EQUAL, 0, "c_Z1_" + std::to_string(e)
        + "_" + std::to_string(j));
    }
    e++;
  }

  // Constraint Z 2
  // Guarantees sample e was not put out of the backbone by attributes in previous nodes (less than j)
  e = 0;
  for (const auto& s : ds.samples_) {
    for (int64_t j = 0; j < bbSize; j++) {
      for (int64_t k = 0; k < j; k++) {
        GRBLinExpr expr;
        expr += Z_[e][j];
        for (int64_t i = 0; i < totAttributes_ - 1; i++) {
          if (s->inxValue_[i] != bbValue_[i]) {
            expr += X_[i][k];
          }
        }
        model_.addConstr(expr, GRB_LESS_EQUAL, 1, "c_Z2_" + std::to_string(e)
          + "_" + std::to_string(j) + "_" + std::to_string(k));
      }
    }
    e++;
  }

  // Constraint W 1
  // Guarantees only one class is chosen to each box
  for (int64_t j = 0; j < bbSize; j++) {
    for (int64_t h = 0; h < maxAttribSize_; h++) {
      GRBLinExpr expr;
      for (int64_t c = 0; c < totClasses_; c++) {
        expr += W_[c][j][h];
      }
      model_.addConstr(expr, GRB_EQUAL, 1, "c_W_" + std::to_string(j) + "_" + std::to_string(h));
    }
  }

  // Constraint V 1
  // V[e][c][j][h] must be limited above by Z[e][j]
  e = 0;
  for (const auto& s : ds.samples_) {
    for (int64_t j = 0; j < bbSize; j++) {
      GRBLinExpr expr;
      for (int64_t c = 0; c < totClasses_; c++) {
        for (int64_t h = 0; h < maxAttribSize_; h++) {
          expr += V_[e][c][j][h];
        }
      }
      expr += -1 * Z_[e][j];
      model_.addConstr(expr, GRB_LESS_EQUAL, 0, "c_V1_" + std::to_string(e)
        + "_c_" + std::to_string(j) + "_h");
    }
    e++;
  }

  // Constraint V 2
  // V[e][c][j][h] must be limited above by W[c][j][h]
  e = 0;
  for (const auto& s : ds.samples_) {
    for (int64_t c = 0; c < totClasses_; c++) {
      for (int64_t j = 0; j < bbSize; j++) {
        for (int64_t h = 0; h < maxAttribSize_; h++) {
          GRBLinExpr expr;
          expr += V_[e][c][j][h];
          expr += -1 * W_[c][j][h];
          model_.addConstr(expr, GRB_LESS_EQUAL, 0, "c_V2_" + std::to_string(e)
            + "_" + std::to_string(c) + "_" + std::to_string(j)
            + "_" + std::to_string(h));
        }
      }
    }
    e++;
  }

  // Constraint V 3
  e = 0;
  for (const auto& s : ds.samples_) {
    for (int64_t j = 0; j < bbSize; j++) {
      for (int64_t h = 0; h < maxAttribSize_; h++) {
        GRBLinExpr expr;
        for (int64_t c = 0; c < totClasses_; c++) {
          expr += V_[e][c][j][h];
        }
        for (int64_t i = 0; i < totAttributes_ - 1; i++) {
          if (s->inxValue_[i] == h) {
            expr += -1 * X_[i][j];
          }
        }
        model_.addConstr(expr, GRB_LESS_EQUAL, 0, "c_V3_" + std::to_string(e)
          + "_c_" + std::to_string(j) + "_" + std::to_string(h));
      }
    }
    e++;
  }

  // Constraint Y 1
  // Sum of all Y_c must be 1
  GRBLinExpr exprY;
  for (int64_t c = 0; c < totClasses_; c++) {
    exprY += Y_[c];
  }
  model_.addConstr(exprY, GRB_EQUAL, 1, "c_Y");

  // Constraint L 1
  // L[e][c] <= Y[c]
  e = 0;
  for (const auto& s : ds.samples_) {
    for (int64_t c = 0; c < totClasses_; c++) {
      GRBLinExpr expr;
      expr += L_[e][c];
      expr += -1 * Y_[c];
      model_.addConstr(expr, GRB_LESS_EQUAL, 0, "c_L1_" + std::to_string(e)
        + "_" + std::to_string(c));
    }
    e++;
  }

  // Constraint L 2
  // L[e][c] > 0 if and only if falls on the backbone on all levels
  for (int64_t j = 0; j < bbSize; j++) {
    e = 0;
    for (const auto& s : ds.samples_) {
      GRBLinExpr expr;
      for (int64_t c = 0; c < totClasses_; c++) {
        expr += L_[e][c];
      }
      for (int64_t i = 0; i < totAttributes_ - 1; i++) {
        if (s->inxValue_[i] != bbValue_[i]) {
          expr += X_[i][j];
        }
      }
      model_.addConstr(expr, GRB_LESS_EQUAL, 1, "c_L2_" + std::to_string(e)
        + "_c_" + std::to_string(j));
      e++;
    }
  }

  // Constraint L 3
  e = 0;
  for (const auto& s : ds.samples_) {
    GRBLinExpr expr;
    for (int64_t c = 0; c < totClasses_; c++) {
      for (int64_t j = 0; j < bbSize; j++) {
        for (int64_t h = 0; h < maxAttribSize_; h++) {
          expr += V_[e][c][j][h];
        }
      }
    }
    for (int64_t c = 0; c < totClasses_; c++) {
      expr += L_[e][c];
    }
    model_.addConstr(expr, GRB_EQUAL, 1, "c_L3_" + std::to_string(e) + "_c");
    e++;
  }
}

int64_t PineTree::getBackBoneValue(DataSet& ds, int64_t col) {
  int64_t attribSize = ds.getAttributeSize(col);
  int64_t ans = 0;
  for (int64_t i = 1; i < attribSize; i++) {
    if (ds.getAttributeFrequency(col, i) > ds.getAttributeFrequency(col, ans)) {
      ans = i;
    }
  }
  return ans;
}

void PineTree::printBB() {
  std::cout << "X values:" << std::endl;
  for (int64_t j = 0; j < X_[0].size(); j++) {
    std::cout << j << " ";
    for (int64_t i = 0; i < totAttributes_ - 1; i++) {
      if (X_[i][j].get(GRB_DoubleAttr_X) > 1e-5) {
        std::cout << "(" << i << ", " << bbValue_[i] << ") ";
      }
    }
    std::cout << std::endl;
  }

  std::cout << "W values:" << std::endl;
  for (int64_t j = 0; j < W_[0].size(); j++) {
    for (int64_t h = 0; h < maxAttribSize_; h++) {
      std::cout << "(" << j << ", " << h << "): ";
      for (int64_t c = 0; c < totClasses_; c++) {
        if (W_[c][j][h].get(GRB_DoubleAttr_X) > 1e-5) {
          std::cout << c << " ";
        }
      }
      std::cout << std::endl;
    }
  }

  std::cout << "V values:" << std::endl;
  for (int64_t e = 0; e < V_.size(); e++) {
    for (int64_t c = 0; c < V_[e].size(); c++) {
      for (int64_t j = 0; j < V_[e][c].size(); j++) {
        for (int64_t h = 0; h < V_[e][c][j].size(); h++) {
          std::cout << "(" << e << ", " << c << ", " << j << ", " << h << "): " << V_[e][c][j][h].get(GRB_DoubleAttr_X) << std::endl;
        }
      }
    }
  }
  std::cout << "L values:" << std::endl;
  for (int64_t e = 0; e < V_.size(); e++) {
    for (int64_t c = 0; c < V_[e].size(); c++) {
      std::cout << "(" << e << ", " << c << "): " << L_[e][c].get(GRB_DoubleAttr_X) << std::endl;
    }
  }
  model_.write("model.lp");
}

void PineTree::defineNodeLeaves(DataSet& ds, int64_t bbSize) {
  initMultidimension(nodeLeaves_, { bbSize, maxAttribSize_ });
  for (int64_t j = 0; j < bbSize; j++) {
    for (int64_t h = 0; h < maxAttribSize_; h++) {
      nodeLeaves_[j][h] = -1;
    }
  }
  nodeAttrib_.resize(bbSize);

  if (varType_ == GRB_INTEGER) {
    defineNodeLeavesInteger(ds, bbSize);
  } else {
    defineNodeLeavesContinuous(ds, bbSize);
  }
}

void PineTree::defineNodeLeavesInteger(DataSet& ds, int64_t bbSize) {
  for (int64_t j = 0; j < bbSize; j++) {
    // Finds the attribute associated with node j
    for (int64_t i = 0; i < totAttributes_ - 1; i++) {
      if (X_[i][j].get(GRB_DoubleAttr_X) > 1e-5) {
        nodeAttrib_[j] = i;
        break;
      }
    }
    // Finds the class of each box of node j
    int64_t attribSize = ds.getAttributeSize(nodeAttrib_[j]);
    for (int64_t h = 0; h < attribSize; h++) {
      if (bbValue_[nodeAttrib_[j]] != h) {
        for (int64_t c = 0; c < totClasses_; c++) {
          if (W_[c][j][h].get(GRB_DoubleAttr_X) > 1e-5) {
            nodeLeaves_[j][h] = c;
            break;
          }
        }
      }
    }
  }
  // Value set to the backbone on the last node
  for (int64_t c = 0; c < totClasses_; c++) {
    if (Y_[c].get(GRB_DoubleAttr_X) > 1e-5) {
      nodeLeaves_[bbSize - 1][bbValue_[nodeAttrib_[bbSize - 1]]] = c;
      break;
    }
  }
}

void PineTree::defineNodeLeavesContinuous(DataSet& ds, int64_t bbSize) {
  // Define which attribute will be matched to each backbone node
  std::set<int64_t> used;
  for (int64_t j = 0; j < bbSize; j++) {
    std::vector<std::pair<double, int64_t>> available;
    for (int64_t i = 0; i < totAttributes_ - 1; i++) {
      if (used.find(i) == used.end()) {
        available.push_back(std::make_pair(X_[i][j].get(GRB_DoubleAttr_X), i));
      }
    }
    std::pair<double, int64_t> best = selectMaxPair(available);
    nodeAttrib_[j] = best.second;
    used.insert(best.second);
  }

  // For each backbone node define the class of each leaf
  for (int64_t j = 0; j < bbSize; j++) {
    int64_t attribSize = ds.getAttributeSize(nodeAttrib_[j]);
    for (int64_t h = 0; h < attribSize; h++) {
      // h == bbValue_[nodeAttrib_[j]] is not a leaf
      if (h != bbValue_[nodeAttrib_[j]]) {
        std::vector<std::pair<double, int64_t>> available;
        for (int64_t c = 0; c < totClasses_; c++) {
          available.push_back(std::make_pair(W_[c][j][h].get(GRB_DoubleAttr_X), c));
        }
        std::pair<double, int64_t> best = selectMaxPair(available);
        nodeLeaves_[j][h] = best.second;
      }
    }
  }

  // For the last node the bbValue is also a leaf
  std::vector<std::pair<double, int64_t>> available;
  for (int64_t c = 0; c < totClasses_; c++) {
    available.push_back(std::make_pair(Y_[c].get(GRB_DoubleAttr_X), c));
  }
  std::pair<double, int64_t> best = selectMaxPair(available);
  nodeLeaves_[bbSize - 1][bbValue_[nodeAttrib_[bbSize - 1]]] = best.second;
}

std::pair<double, int64_t> PineTree::selectMaxPair(std::vector<std::pair<double, int64_t>>& v) {
  ErrorUtils::enforce(v.size() > 0, "Array must have at least one element.");
  std::pair<double, int64_t> best(v[0]);
  for (int64_t i = 1; i < v.size(); i++) {
    if (v[i].first > best.first) {
      best = v[i];
    }
  }
  return best;
}

std::shared_ptr<DecisionTreeNode> PineTree::mountBackbone(DataSet& ds, int64_t bbSize) {
  return mountBackboneLevel(ds, 0, bbSize);
}

std::shared_ptr<DecisionTreeNode> PineTree::mountBackboneLevel(DataSet& ds, int64_t level, int64_t bbSize) {
  std::shared_ptr<DecisionTreeNode> node = std::make_shared<DecisionTreeNode>(DecisionTreeNode::NodeType::REGULAR, nodeAttrib_[level]);
  node->setName("(" + std::to_string(level) + ", " + std::to_string(nodeAttrib_[level]) + ")");
  for (int64_t h = 0; h < nodeLeaves_[level].size(); h++) {
    if (nodeLeaves_[level][h] != -1) {
      std::shared_ptr<DecisionTreeNode> leaf = std::make_shared<DecisionTreeNode>(DecisionTreeNode::NodeType::LEAF);
      leaf->setLeafValue(nodeLeaves_[level][h]);
      node->addChild(leaf, { h });
    }
  }

  if (level < bbSize - 1) {
    node->addChild(mountBackboneLevel(ds, level + 1, bbSize), { bbValue_[nodeAttrib_[level]] });
  }
  return node;
}

std::pair<int64_t, std::shared_ptr<DecisionTreeNode>> PineTree::getNextNodeBB(std::shared_ptr<DecisionTreeNode> curr) {
  int64_t nonLeaf = 0;
  std::shared_ptr<DecisionTreeNode> ans = nullptr;
  int64_t inx;
  for (const auto& child : curr->children_) {
    if (!child.second->isLeaf()) {
      nonLeaf++;
      ans = child.second;
      inx = child.first;
    }
  }
  if (nonLeaf == 1) {
    return std::pair<int64_t, std::shared_ptr<DecisionTreeNode>>(inx, ans);
  }
  return std::pair<int64_t, std::shared_ptr<DecisionTreeNode>>(-1, nullptr);
}
