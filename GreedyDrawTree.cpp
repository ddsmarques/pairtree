#include "GreedyDrawTree.h"

#include "CompareUtils.h"

#include <random>


std::shared_ptr<DecisionTreeNode> GreedyDrawTree::createTree(DataSet& ds, std::shared_ptr<ConfigTree> c) {
  std::shared_ptr<ConfigGreedyDraw> config = std::static_pointer_cast<ConfigGreedyDraw>(c);
  std::vector<bool> availableAttrib(ds.getTotAttributes(), true);
  return createTreeRec(ds, config->height, config->totDraws, availableAttrib);
}


std::shared_ptr<DecisionTreeNode> GreedyDrawTree::createTreeRec(DataSet& ds,
                                                                int64_t height, int totDraws,
                                                                std::vector<bool> availableAttrib) {
  ErrorUtils::enforce(ds.getTotClasses() > 0, "Invalid data set.");

  if (height == 0) {
    return createLeaf(ds);
  }

  int64_t bestAttrib = -1;
  for (int64_t i = 0; i < ds.getTotAttributes(); i++) {
    if (availableAttrib[i] && isGoodAttribute(ds, i, totDraws)) {
      bestAttrib = i;
      break;
    }
  }
  // There are no good attributes left
  if (bestAttrib == -1) {
    return createLeaf(ds);
  }
  availableAttrib[bestAttrib] = false;

  int64_t bestAttribSize = ds.getAttributeSize(bestAttrib);
  std::vector<DataSet> allDS(bestAttribSize);
  for (int64_t i = 0; i < bestAttribSize; i++) {
    allDS[i].initAllAttributes(ds);
  }
  for (const auto& s : ds.samples_) {
    allDS[s->inxValue_[bestAttrib]].addSample(s);
  }

  std::shared_ptr<DecisionTreeNode> node = std::make_shared<DecisionTreeNode>(DecisionTreeNode::NodeType::REGULAR, bestAttrib);
  for (int64_t j = 0; j < bestAttribSize; j++) {
    node->addChild(createTreeRec(allDS[j], height - 1, totDraws, availableAttrib), { j });
  }
  return node;
}


bool GreedyDrawTree::isGoodAttribute(DataSet& ds, int64_t attribInx, int totDraws) {
  double attribScore = 0;
  for (int64_t i = 0; i < ds.getAttributeSize(attribInx); i++) {
    auto bestClass = ds.getSubDataSet(attribInx, i).getBestClass();
    attribScore += bestClass.second;
  }

  for (int i = 0; i < totDraws; i++) {
    double score = getRandomAttribute(ds, attribInx);
    if (CompareUtils::compare(attribScore, score) < 0) {
      return false;
    }
  }
  return true;
}


double GreedyDrawTree::getRandomAttribute(DataSet& ds, int64_t attribInx) {
  // distrib[i] = number of samples valued 'i' for this attribute
  std::vector<int64_t> distrib(ds.getAttributeSize(attribInx), 0);
  for (auto s : ds.samples_) {
    distrib[s->inxValue_[attribInx]]++;
  }

  // Copies the samples to a vector. This is needed to run the shuffling
  std::vector<std::shared_ptr<Sample>> samplesShuffle(ds.samples_.size());
  auto it = ds.samples_.begin();
  for (int64_t i = 0; i < samplesShuffle.size(); i++) {
    samplesShuffle[i] = *it;
    it++;
  }
  std::random_device rd;
  std::mt19937 gen(rd());
  std::shuffle(samplesShuffle.begin(), samplesShuffle.end(), gen);

  double score = 0;
  int64_t sInx = 0; // sample index
  for (int j = 0; j < distrib.size(); j++) {
    // Calculates the best class for the distrib[j] samples on the same leaf
    std::vector<double> classBenefit(ds.getTotClasses(), 0);
    for (int64_t i = 0; i < distrib[j]; i++) {
      for (int k = 0; k < ds.getTotClasses(); k++) {
        classBenefit[k] += samplesShuffle[sInx]->benefit_[k];
      }
      sInx++;
    }
    double bestClass = classBenefit[0];
    for (int k = 1; k < classBenefit.size(); k++) {
      if (CompareUtils::compare(bestClass, classBenefit[k]) < 0) {
        bestClass = classBenefit[k];
      }
    }
    score += bestClass;
  }
  return score;
}

std::shared_ptr<DecisionTreeNode> GreedyDrawTree::createLeaf(DataSet& ds) {
  auto best = ds.getBestClass();

  std::shared_ptr<DecisionTreeNode> leaf = std::make_shared<DecisionTreeNode>(DecisionTreeNode::NodeType::LEAF);
  leaf->setName("LEAF " + std::to_string(best.first));
  leaf->setLeafValue(best.first);

  return leaf;
}
