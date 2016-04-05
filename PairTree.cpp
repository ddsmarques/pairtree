#include "PairTree.h"

#include "CompareUtils.h"
#include "Logger.h"


std::shared_ptr<DecisionTreeNode> PairTree::createTree(DataSet& ds, std::shared_ptr<ConfigTree> c) {
  ErrorUtils::enforce(ds.getTotClasses() == 2, "Error! Number of classes must be 2.");
  
  std::shared_ptr<ConfigPairTree> config = std::static_pointer_cast<ConfigPairTree>(c);
  return createTreeRec(ds, config->height);
}


std::shared_ptr<DecisionTreeNode> PairTree::createTreeRec(DataSet& ds, int height) {
  if (height == 0) {
    return createLeaf(ds);
  }

  std::vector<SampleInfo> samplesInfo;
  initSampleInfo(ds, samplesInfo);

  int bestAttrib = -1;
  double bestScoreRatio = 0;
  for (int64_t i = 0; i < ds.getTotAttributes(); i++) {
    long double score = getAttribScore(ds, i, samplesInfo);
    auto randScore = getRandomScore(ds, i, samplesInfo);
    long double expected = randScore.first;
    if (CompareUtils::compare(expected, 0) != 0
        && CompareUtils::compare(score / expected, 1) > 0
        && CompareUtils::compare(score / expected, bestScoreRatio) > 0) {
      bestAttrib = i;
      bestScoreRatio = score / expected;
    }
  }

  if (bestAttrib == -1) {
    return createLeaf(ds);
  }

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
    node->addChild(createTreeRec(allDS[j], height - 1), { j });
  }
  return node;
}


long double PairTree::getAttribScore(DataSet& ds, int64_t attribInx, std::vector<PairTree::SampleInfo>& samplesInfo) {
  int64_t attribSize = ds.getAttributeSize(attribInx);
  std::vector<int64_t> totalClass(2, 0); // totalClass[0] = number of samples whose best class is 0
  // totalValueClass[j][c] = number of samples valued 'j' at 'attribInx' whose class is 'c'
  std::vector<std::vector<int64_t>> totalValueClass(attribSize, std::vector<int64_t>(2, 0));

  for (auto s : samplesInfo) {
    totalClass[s.bestClass]++;
    totalValueClass[s.ptr->inxValue_[attribInx]][s.bestClass]++;
  }
  int64_t totPairs = totalClass[0] * totalClass[1];

  long double score = 0;
  for (auto s : samplesInfo) {
    int notBestClass = (s.bestClass + 1) % 2;
    score += s.diff * (totalClass[notBestClass] - totalValueClass[s.ptr->inxValue_[attribInx]][notBestClass]);
    totalClass[s.bestClass]--;
    totalValueClass[s.ptr->inxValue_[attribInx]][s.bestClass]--;
  }
  return score / totPairs;
}


std::pair<long double, long double> PairTree::getRandomScore(DataSet& ds, int64_t attribInx, std::vector<PairTree::SampleInfo>& samplesInfo) {
  int64_t attribSize = ds.getAttributeSize(attribInx);
  std::vector<int64_t> totalClass(2, 0); // totalClass[0] = number of samples whose best class is 0
  std::vector<double> probValue(attribSize, 0); // probValue[j] = probability of a sample be valued 'j' in attribute attribInx
  for (auto s : samplesInfo) {
    totalClass[s.bestClass]++;
    probValue[s.ptr->inxValue_[attribInx]]++;
  }
  for (int j = 0; j < attribSize; j++) {
    probValue[j] = probValue[j] / samplesInfo.size();
  }
  int64_t totPairs = totalClass[0] * totalClass[1];
  std::vector<int64_t> totalClassCopy = totalClass;


  long double expected = 0;
  for (auto s : samplesInfo) {
    int notBestClass = (s.bestClass + 1) % 2;
    for (int j = 0; j < attribSize; j++) {
      double p = probValue[j];
      expected += s.diff * totalClass[notBestClass] * p * (1 - p);
    }
    totalClass[s.bestClass]--;
  }
  expected = expected /totPairs;
  
  totalClass = totalClassCopy;
  long double var = 0;
  for (auto s : samplesInfo) {
    int notBestClass = (s.bestClass + 1) % 2;
    for (int j = 0; j < attribSize; j++) {
      double p = probValue[j];
      var += (((long double)s.diff) - expected) * (((long double)s.diff) - expected) * totalClass[notBestClass] * p * (1 - p);
    }
    totalClass[s.bestClass]--;
  }
  var = var / totPairs;
  long double std = sqrt(var);

  return std::make_pair(expected, std);
}

bool compareSampleInfo(const PairTree::SampleInfo& a, const PairTree::SampleInfo& b) {
  return CompareUtils::compare(a.diff, b.diff) < 0;
}

void PairTree::initSampleInfo(DataSet& ds, std::vector<PairTree::SampleInfo>& samplesInfo) {
  samplesInfo.resize(ds.samples_.size());
  int64_t count = 0;
  for (auto s : ds.samples_) {
    SampleInfo info;
    info.ptr = s;
    if (CompareUtils::compare(s->benefit_[0], s->benefit_[1]) >= 0) {
      info.bestClass = 0;
    }
    else {
      info.bestClass = 1;
    }
    info.diff = std::abs(s->benefit_[0] - s->benefit_[1]);

    samplesInfo[count++] = info;
  }
  std::sort(samplesInfo.begin(), samplesInfo.end(), compareSampleInfo);
}


std::shared_ptr<DecisionTreeNode> PairTree::createLeaf(DataSet& ds) {
  auto best = ds.getBestClass();

  std::shared_ptr<DecisionTreeNode> leaf = std::make_shared<DecisionTreeNode>(DecisionTreeNode::NodeType::LEAF);
  leaf->setName("LEAF " + std::to_string(best.first));
  leaf->setLeafValue(best.first);

  return leaf;
}
