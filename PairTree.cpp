#include "PairTree.h"

#include "Attribute.h"
#include "BIT.h"
#include "CompareUtils.h"
#include "Logger.h"
#include "ExtrasTreeNode.h"

#include <cmath>


std::shared_ptr<DecisionTreeNode> PairTree::createTree(DataSet& ds, std::shared_ptr<ConfigTree> c) {
  ErrorUtils::enforce(ds.getTotClasses() == 2, "Error! Number of classes must be 2.");
  
  std::shared_ptr<ConfigPairTree> config = std::static_pointer_cast<ConfigPairTree>(c);
  BoundType boundOption;
  if (config->boundOption.compare("DIFF") == 0) {
    boundOption = BoundType::DIFF_BOUND;
  }
  else if (config->boundOption.compare("T") == 0) {
    boundOption = BoundType::T_BOUND;
  }
  else if (config->boundOption.compare("VAR") == 0) {
    boundOption = BoundType::VAR_BOUND;
  }
  else {
    Logger::log() << "Wrong boundOption for PairTree. Using default DIFF_BOUND value.";
    boundOption = BoundType::DIFF_BOUND;
  }
  return createTreeRec(ds, config->height, config->maxBound, config->minLeaf,
                       config->useScore, config->useNominalBinary, boundOption);
}


std::shared_ptr<DecisionTreeNode> PairTree::createTreeRec(DataSet& ds, int height,
                                                          double maxBound,
                                                          int64_t minLeaf, bool useScore,
                                                          bool useNominalBinary,
                                                          BoundType boundType) {
  if (height == 0 || (minLeaf > 0 && ds.samples_.size() <= minLeaf)
      || isAllSameClass(ds)) {
    return createLeaf(ds);
  }

  std::vector<SampleInfo> samplesInfo;
  initSampleInfo(ds, samplesInfo);

  int bestAttrib = -1;
  double bestBound = 1;
  int64_t bestSeparator = -1;
  long double bestScore = 0;
  for (int64_t i = 0; i < ds.getTotAttributes(); i++) {
    auto attribResult = testAttribute(ds, i, samplesInfo, useNominalBinary, boundType);
    // If bound satisfy maxBound then gets either greatest score or lowest bound
    if (CompareUtils::compare(attribResult.bound, maxBound) < 0
        && ((useScore && CompareUtils::compare(attribResult.score, bestScore) > 0)
            || (!useScore && CompareUtils::compare(attribResult.bound, bestBound) < 0))) {
      bestAttrib = i;
      bestBound = attribResult.bound;
      bestSeparator = attribResult.separator;
      bestScore = attribResult.score;
    }
  }

  if (bestAttrib == -1) {
    return createLeaf(ds);
  }

  // Nominal k-valued attribute creating k children
  if (bestSeparator == -1) {
    int64_t bestAttribSize = ds.getAttributeSize(bestAttrib);
    std::vector<DataSet> allDS(bestAttribSize);
    for (int64_t i = 0; i < bestAttribSize; i++) {
      allDS[i].initAllAttributes(ds);
    }
    for (const auto& s : ds.samples_) {
      allDS[s->inxValue_[bestAttrib]].addSample(s);
    }

    std::shared_ptr<ExtrasTreeNode> node = std::make_shared<ExtrasTreeNode>(DecisionTreeNode::NodeType::REGULAR_NOMINAL, bestAttrib);
    node->setAlpha(bestBound);
    node->setNumSamples(ds.samples_.size());
    node->setLeafValue(ds.getBestClass().first);
    for (int64_t j = 0; j < bestAttribSize; j++) {
      node->addChild(createTreeRec(allDS[j], height - 1, maxBound, minLeaf, useScore, useNominalBinary, boundType), { j });
    }
    return node;

  // Nominal attribute separating one value from the rest
  } else if (bestSeparator != -1 && ds.getAttributeType(bestAttrib) == AttributeType::STRING) {
    std::shared_ptr<ExtrasTreeNode> node = std::make_shared<ExtrasTreeNode>(DecisionTreeNode::NodeType::REGULAR_NOMINAL, bestAttrib);
    node->setAlpha(bestBound);
    node->setNumSamples(ds.samples_.size());
    node->setLeafValue(ds.getBestClass().first);

    DataSet leftDS, rightDS;
    leftDS.initAllAttributes(ds);
    rightDS.initAllAttributes(ds);
    for (const auto& s : ds.samples_) {
      if (s->inxValue_[bestAttrib] == bestSeparator) {
        leftDS.addSample(s);
      } else {
        rightDS.addSample(s);
      }
    }
    
    node->addChild(createTreeRec(leftDS, height - 1, maxBound, minLeaf, useScore, useNominalBinary, boundType), { bestSeparator });
    std::vector<int64_t> rightInxs;
    for (int64_t i = 0; i < ds.getAttributeSize(bestAttrib); i++) {
      if (i != bestSeparator) rightInxs.push_back(i);
    }
    node->addChild(createTreeRec(rightDS, height - 1, maxBound, minLeaf, useScore, useNominalBinary, boundType), rightInxs);

    return node;

  // Numeric Attribute
  } else {
    std::shared_ptr<ExtrasTreeNode> node = std::make_shared<ExtrasTreeNode>(DecisionTreeNode::NodeType::REGULAR_ORDERED, bestAttrib, bestSeparator);
    node->setAlpha(bestBound);
    node->setNumSamples(ds.samples_.size());
    node->setLeafValue(ds.getBestClass().first);
    DataSet leftDS, rightDS;
    leftDS.initAllAttributes(ds);
    rightDS.initAllAttributes(ds);
    for (const auto& s : ds.samples_) {
      if (s->inxValue_[bestAttrib] <= bestSeparator) {
        leftDS.addSample(s);
      } else {
        rightDS.addSample(s);
      }
    }
    node->addLeftChild(createTreeRec(leftDS, height - 1, maxBound, minLeaf, useScore, useNominalBinary, boundType));
    node->addRightChild(createTreeRec(rightDS, height - 1, maxBound, minLeaf, useScore, useNominalBinary, boundType));
    return node;
  }
}


std::shared_ptr<DecisionTreeNode> PairTree::createLeaf(DataSet& ds) {
  auto best = ds.getBestClass();

  std::shared_ptr<ExtrasTreeNode> leaf = std::make_shared<ExtrasTreeNode>(DecisionTreeNode::NodeType::LEAF);
  leaf->setName("LEAF " + std::to_string(best.first));
  leaf->setLeafValue(best.first);

  return leaf;
}


bool PairTree::isAllSameClass(DataSet& ds) {
  int64_t tot0 = 0;
  int64_t tot1 = 0;
  for (auto s : ds.samples_) {
    if (CompareUtils::compare(s->benefit_[0], s->benefit_[1]) >= 0) {
      tot0++;
    }
    else {
      tot1++;
    }
    if (tot0 > 0 && tot1 > 0) return false;
  }
  return tot0 == 0 || tot1 == 0;
}


PairTree::AttribResult PairTree::testAttribute(DataSet& ds, int64_t attribInx,
                                               std::vector<PairTree::SampleInfo>& samplesInfo,
                                               bool useNominalBinary,
                                               BoundType boundType) {
  if (ds.getAttributeType(attribInx) == AttributeType::INTEGER
      || ds.getAttributeType(attribInx) == AttributeType::DOUBLE) {
    return testNumeric(ds, attribInx, samplesInfo, boundType);
  } else {
    return testNominal(ds, attribInx, samplesInfo, useNominalBinary, boundType);
  }
}


PairTree::AttribResult PairTree::testNominal(DataSet& ds, int64_t attribInx,
                                             std::vector<PairTree::SampleInfo>& samplesInfo,
                                             bool useNominalBinary,
                                             BoundType boundType) {
  AttribResult best;
  best.bound = 1;

  int64_t attribSize = ds.getAttributeSize(attribInx);
  // If wants to use binary splits then should test from -1 to attribSize.
  // Else only needs to test boxZero = -1
  int64_t maxSplits = useNominalBinary ? attribSize : 0;
  int64_t startSplit = useNominalBinary ? 0 : -1;
  for (int64_t i = startSplit; i < maxSplits; i++) {
    int64_t boxZero = i;
    std::function<int64_t(int64_t)> valueBox = [boxZero](int64_t inxValue) {
      if (boxZero < 0) {
        return inxValue;
      }
      return inxValue == boxZero ? 0ll : 1ll;
    };

    auto scoreResult = calcNominalScore(ds, attribInx, valueBox, boxZero == -1 ? attribSize : 2, samplesInfo);
    long double bound = getAttribBound(ds, scoreResult, attribInx, samplesInfo, boundType);
    long double diffBound = getAttribBound(ds, scoreResult, attribInx, samplesInfo, BoundType::DIFF_BOUND);
    bound = std::min(bound, diffBound);
    if (CompareUtils::compare(bound, best.bound) < 0) {
      best.bound = bound;
      best.score = scoreResult.score;
      best.separator = boxZero;
    }
  }

  return best;
}


PairTree::AttribScoreResult PairTree::calcNominalScore(DataSet& ds, int64_t attribInx,
                                                       std::function<int64_t(int64_t)> valueBox,
                                                       int64_t attribSize,
                                                       std::vector<PairTree::SampleInfo>& samplesInfo) {
  std::vector<double> distrib(attribSize);
  for (int64_t i = 0; i < samplesInfo.size(); i++) {
    distrib[valueBox(samplesInfo[i].ptr->inxValue_[attribInx])]++;
  }
  for (int64_t j = 0; j < attribSize; j++) {
    distrib[j] = distrib[j] / samplesInfo.size();
  }

  // totalClass[0] = number of samples whose best class is 0
  std::vector<int64_t> totalClass(2, 0);
  // totalValueClass[j][c] = number of samples valued 'j' at 'attribInx' whose class is 'c'
  std::vector<std::vector<int64_t>> totalValueClass(attribSize, std::vector<int64_t>(2, 0));

  for (auto s : samplesInfo) {
    totalClass[s.bestClass]++;
    totalValueClass[valueBox(s.ptr->inxValue_[attribInx])][s.bestClass]++;
  }
  int64_t totPairs = totalClass[0] * totalClass[1];

  long double score = 0;
  for (auto s : samplesInfo) {
    int notBestClass = (s.bestClass + 1) % 2;
    score += s.diff * (totalClass[notBestClass] - totalValueClass[valueBox(s.ptr->inxValue_[attribInx])][notBestClass]);
    totalClass[s.bestClass]--;
    totalValueClass[valueBox(s.ptr->inxValue_[attribInx])][s.bestClass]--;
  }

  AttribScoreResult ans;
  ans.score = score;
  ans.distrib = distrib;
  ans.separator = -1;
  return ans;
}


PairTree::AttribResult PairTree::testNumeric(DataSet& ds, int64_t attribInx,
                                             std::vector<PairTree::SampleInfo>& samplesInfo,
                                             BoundType boundType) {
  // Calculate the sum on formula E[Gain(A)] = 2*p(1-p) * \sum_{i=1...N}{D(s_i) * TC^i_{notC}}
  // The following variables will be used later to calculate the bound for a splitting parameter
  auto randomScore = getRandomScore(samplesInfo, std::vector<double>{ 0.5, 0.5 });
  long double randomSum = 2 * randomScore.first;
  BoundConstants constants;
  BoundConstants extraConstants;
  long double sumDSq;
  long double maxG;
  
  if (boundType != BoundType::VAR_BOUND) {
    constants = calcConstants(ds, attribInx, samplesInfo, boundType);
  }
  else {
    sumDSq = calcConstSumDSq(attribInx, samplesInfo);
    maxG = calcMaxD(ds);
    constants.xstar = calcConstXstar(ds);
    extraConstants = calcConstants(ds, attribInx, samplesInfo, BoundType::DIFF_BOUND);
  }

  long double bestBound = 1;
  long double bestScore = 0;
  int64_t bestLeftSize = 0;
  int64_t bestSeparator = 0;
  int64_t totSamples = ds.samples_.size();

  // Sort all samples by attribute attribInx
  struct Order {
    int64_t attribValue;
    int64_t posDiff;
  };
  std::vector<Order> ordSamples(totSamples);
  for (int64_t i = 0; i < samplesInfo.size(); i++) {
    Order aux;
    aux.posDiff = i;
    aux.attribValue = samplesInfo[i].ptr->inxValue_[attribInx];
    ordSamples[i] = aux;
  }
  std::sort(ordSamples.begin(), ordSamples.end(), [](const Order& a, const Order& b) { return a.attribValue < b.attribValue; });

  std::vector<std::vector<BIT>> sumLeft(2, std::vector<BIT>(2, totSamples));
  std::vector<BIT> countLeft(2, totSamples);
  std::vector<std::vector<BIT>> sumRight(2, std::vector<BIT>(2, totSamples));
  std::vector<BIT> countRight(2, totSamples);

  // Put all samples to the right
  for (int64_t i = 0; i < ordSamples.size(); i++) {
    int64_t posDiff = ordSamples[i].posDiff;
    int64_t bestClass = samplesInfo[posDiff].bestClass;
    countRight[bestClass].update(posDiff + 1, 1);
    sumRight[bestClass][0].update(posDiff + 1, samplesInfo[posDiff].ptr->benefit_[0]);
    sumRight[bestClass][1].update(posDiff + 1, samplesInfo[posDiff].ptr->benefit_[1]);
  }
  
  long double score = 0;
  int64_t i = 0;
  for (int64_t i = 0; i < totSamples; i++) {
    int64_t posDiff = ordSamples[i].posDiff;
    int bestClass = samplesInfo[posDiff].bestClass;
    int worstClass = (bestClass + 1) % 2;

    // First part max{B(x,0), B(x,1)} + max{B(y,0), B(y,1)}
    score -= (countLeft[worstClass].get(totSamples) * samplesInfo[posDiff].ptr->benefit_[bestClass]
              + sumLeft[worstClass][worstClass].get(totSamples));
    score += (countRight[worstClass].get(totSamples) * samplesInfo[posDiff].ptr->benefit_[bestClass]
              + sumRight[worstClass][worstClass].get(totSamples));

    // Second part
    // C1 = {p | p in S_notC AND D(p) < D(s) AND Ai(p) < Ai(s)}
    //ans -= countLeft[notC].get(i - 1) * B(s, c)
    //ans -= sumLeft[notC][c].get(i - 1)
    score += countLeft[worstClass].get(posDiff) * samplesInfo[posDiff].ptr->benefit_[bestClass];
    score += sumLeft[worstClass][bestClass].get(posDiff);

    // C2 = {p | p in S_notC AND D(p) > D(s) AND Ai(p) < Ai(s)}
    //ans -= (countLeft[notC].get(N) - countLeft[notC].get(i)) * B(s, notC)
    //ans -= (sumLeft[notC][notC].get(N) - sumLeft[notC][notC].get(i))
    score += (countLeft[worstClass].get(totSamples) - countLeft[worstClass].get(posDiff+1)) * samplesInfo[posDiff].ptr->benefit_[worstClass];
    score += (sumLeft[worstClass][worstClass].get(totSamples) - sumLeft[worstClass][worstClass].get(posDiff+1));

    // C3 = {p | p in S_notC AND D(p) < D(s) AND Ai(p) > Ai(s)}
    //ans += countRight[notC].get(i - 1) * B(s, c)
    //ans += sumRight[notC][c].get(i - 1)
    score -= countRight[worstClass].get(posDiff) * samplesInfo[posDiff].ptr->benefit_[bestClass];
    score -= sumRight[worstClass][bestClass].get(posDiff);

    // C4 = {p | p in S_notC AND D(p) > D(s) AND Ai(p) > Ai(s)}
    //ans += (countRight[notC].get(N) - countRight[notC].get(i)) * B(s, notC)
    //ans += (sumRight[notC][notC].get(N) - sumRight[notC][notC].get(i))
    score -= (countRight[worstClass].get(totSamples) - countRight[worstClass].get(posDiff+1)) * samplesInfo[posDiff].ptr->benefit_[worstClass];
    score -= (sumRight[worstClass][worstClass].get(totSamples) - sumRight[worstClass][worstClass].get(posDiff+1));

    sumLeft[bestClass][0].update(posDiff + 1, samplesInfo[posDiff].ptr->benefit_[0]);
    sumLeft[bestClass][1].update(posDiff + 1, samplesInfo[posDiff].ptr->benefit_[1]);
    countLeft[bestClass].update(posDiff + 1, 1);
    sumRight[bestClass][0].update(posDiff + 1, -samplesInfo[posDiff].ptr->benefit_[0]);
    sumRight[bestClass][1].update(posDiff + 1, -samplesInfo[posDiff].ptr->benefit_[1]);
    countRight[bestClass].update(posDiff + 1, -1);

    if (i == totSamples - 1 || (ordSamples[i].attribValue != ordSamples[i + 1].attribValue)) {
      long double p = (i + 1) / ((long double)totSamples);
      long double expected = 2 * p * (1 - p) * randomSum;
      long double bound;

      if (boundType != BoundType::VAR_BOUND) {
        bound = applyBound(score - expected, constants, boundType);
      }
      else {
        long double splitProb = 2 * p * (1 - p);
        constants.S = sumDSq * (splitProb - splitProb * splitProb);
        constants.b = maxG * (1 - splitProb);

        bound = applyBound(score - expected, constants, BoundType::VAR_BOUND);
        long double diffBound = applyBound(score - expected, extraConstants, BoundType::DIFF_BOUND);
        bound = std::min(bound, diffBound);
      }
      if (CompareUtils::compare(bound, bestBound) < 0) {
        bestSeparator = ordSamples[i].attribValue;
        bestBound = bound;
        bestScore = score;
        bestLeftSize = i + 1;
      }
    }
  }

  AttribResult ans;
  ans.score = bestScore;
  ans.bound = bestBound;
  ans.separator = bestSeparator;

  return ans;
}


std::pair<long double, long double> PairTree::getRandomScore(std::vector<PairTree::SampleInfo>& samplesInfo,
                                                             std::vector<double>& distrib) {
  int64_t attribSize = distrib.size();
  std::vector<int64_t> totalClass(2, 0); // totalClass[0] = number of samples whose best class is 0
  for (auto s : samplesInfo) {
    totalClass[s.bestClass]++;
  }
  int64_t totPairs = totalClass[0] * totalClass[1];
  std::vector<int64_t> totalClassCopy = totalClass;

  long double expected = 0;
  for (auto s : samplesInfo) {
    int notBestClass = (s.bestClass + 1) % 2;
    for (int j = 0; j < attribSize; j++) {
      double p = distrib[j];
      expected += s.diff * totalClass[notBestClass] * p * (1 - p);
    }
    totalClass[s.bestClass]--;
  }
  expected = expected / totPairs;

  totalClass = totalClassCopy;
  long double var = 0;
  for (auto s : samplesInfo) {
    int notBestClass = (s.bestClass + 1) % 2;
    for (int j = 0; j < attribSize; j++) {
      double p = distrib[j];
      var += (((long double)s.diff) - expected) * (((long double)s.diff) - expected) * totalClass[notBestClass] * p * (1 - p);
    }
    totalClass[s.bestClass]--;
  }
  var = var / totPairs;
  long double std = sqrt(var);

  return std::make_pair(expected*totPairs, std);
}


long double PairTree::getAttribBound(DataSet& ds, AttribScoreResult& attribResult,
                                     int64_t attribInx,
                                     std::vector<PairTree::SampleInfo>& samplesInfo,
                                     BoundType boundType) {
  auto aux = getRandomScore(samplesInfo, attribResult.distrib);
  long double expected = aux.first;
  if (CompareUtils::compare(attribResult.score, expected, 1e-7) > 0) {
    BoundConstants constants = calcConstants(ds, attribInx, samplesInfo, boundType);
    return applyBound(attribResult.score - expected, constants, boundType);
  }
  return 1;
}


long double PairTree::applyBound(long double t, BoundConstants constants, BoundType boundType) {
  // The bounds only work for t > 0
  if (CompareUtils::compare(t, 0) <= 0) {
    return 1;
  }

  if (boundType == BoundType::DIFF_BOUND) {
    if (CompareUtils::compare(constants.xstar * constants.sumDSq, 0) == 0) {
      return 1;
    }
    return std::exp((-2.0 * t * t) / (constants.xstar * constants.sumDSq));
  }
  else if (boundType == BoundType::T_BOUND) {
    if (CompareUtils::compare(constants.TSq, 0) == 0) {
      return 1;
    }
    return std::exp((-2.0 * t * t) / constants.TSq);
  }
  else if (boundType == BoundType::VAR_BOUND) {
    long double denominator = constants.b * constants.b * constants.xstar;
    if (CompareUtils::compare(denominator, 0) == 0
        || CompareUtils::compare(constants.S, 0) == 0) {
      return 1;
    }
    long long x = 4 * constants.b * t / (5 * constants.S);
    long double phi = (1 + x) * std::log(1 + x) - x;
    return std::exp(-constants.S*phi / denominator);
  }
}


PairTree::BoundConstants PairTree::calcConstants(DataSet& ds, int64_t attribInx,
                                                 std::vector<PairTree::SampleInfo>& samplesInfo,
                                                 BoundType boundType) {
  BoundConstants ans;
  if (boundType == BoundType::DIFF_BOUND) {
    ans.xstar = calcConstXstar(ds);
    ans.sumDSq = calcConstSumDSq(attribInx, samplesInfo);
  }
  else if (boundType == BoundType::T_BOUND) {
    ans.TSq = calcConstTSq(ds);
  }
  else if (boundType == BoundType::VAR_BOUND) {
    ans.xstar = calcConstXstar(ds);
    ans.S = calcConstS(ds, attribInx, samplesInfo);
    ans.b = calcConstb(ds, attribInx);
  }
  return ans;
}


long double PairTree::calcConstXstar(DataSet& ds) {
  int64_t best0 = 0;
  int64_t best1 = 0;
  for (auto s : ds.samples_) {
    if (CompareUtils::compare(s->benefit_[0], s->benefit_[1]) > 0) {
      best0++;
    }
    else {
      best1++;
    }
  }
  return std::max(best0, best1);
}


long double PairTree::calcConstSumDSq(int64_t attribInx,
                                      std::vector<PairTree::SampleInfo>& samplesInfo) {
  std::vector<int64_t> totalClass(2, 0); // totalClass[0] = number of samples whose best class is 0

  for (auto s : samplesInfo) {
    totalClass[s.bestClass]++;
  }

  long double sumDSq = 0;
  for (auto s : samplesInfo) {
    int notBestClass = (s.bestClass + 1) % 2;
    sumDSq += (s.diff * s.diff) * totalClass[notBestClass];
    totalClass[s.bestClass]--;
  }

  return sumDSq;
}


long double PairTree::calcConstTSq(DataSet& ds) {
  std::vector<long double> s0;
  std::vector<long double> s1;
  createTwoDiffs(ds, s0, s1);

  if (s0.size() < s1.size()) {
    std::sort(s0.begin(), s0.end(), std::greater<long double>());
    std::sort(s1.begin(), s1.end(), std::less<long double>());
    return calcMatchingSums(s0, s1);
  }
  else {
    std::sort(s0.begin(), s0.end(), std::less<long double>());
    std::sort(s1.begin(), s1.end(), std::greater<long double>());
    return calcMatchingSums(s1, s0);
  }
}


long double PairTree::calcConstS(DataSet& ds, int64_t attribInx,
                                 std::vector<PairTree::SampleInfo>& samplesInfo) {
  long double sumVarSq = calcConstSumDSq(attribInx, samplesInfo);
  long double splitProb = calcSplitProb(ds, attribInx);
  
  return sumVarSq * splitProb - sumVarSq * splitProb * splitProb;
}


long double PairTree::calcConstb(DataSet& ds, int64_t attribInx) {
  long double maxD = calcMaxD(ds);
  long double splitProb = calcSplitProb(ds, attribInx);

  return maxD * (1 - splitProb);
}


long double PairTree::calcMaxD(DataSet& ds) {
  long double maxBest0 = std::numeric_limits<long double>::min();
  long double maxBest1 = std::numeric_limits<long double>::min();
  for (auto s : ds.samples_) {
    if (CompareUtils::compare(s->benefit_[0], s->benefit_[1]) > 0) {
      maxBest0 = std::max(maxBest0, (long double)s->benefit_[0] - s->benefit_[1]);
    }
    else {
      maxBest1 = std::max(maxBest1, (long double)s->benefit_[1] - s->benefit_[0]);
    }
  }
  
  return std::min(maxBest0, maxBest1);
}


long double PairTree::calcSplitProb(DataSet& ds, int64_t attribInx) {
  std::vector<long double> freqs = ds.getAttributeCurrentFullFrequency(attribInx);
  long double splitProb = 0;
  for (int64_t i = 0; i < ds.getAttributeSize(attribInx); i++) {
    splitProb += freqs[i] * (1 - freqs[i]);
  }

  return splitProb;
}


void PairTree::createTwoDiffs(DataSet& ds, std::vector<long double>& s0,
                              std::vector<long double>& s1) {
  int64_t totS0 = 0;
  for (auto& s : ds.samples_) {
    if (CompareUtils::compare(s->benefit_[0], s->benefit_[1]) > 0) {
      totS0++;
    }
  }
  int64_t totS1 = ds.samples_.size() - totS0;
  s0.resize(totS0);
  s1.resize(totS1);
  int64_t countS0 = 0;
  int64_t countS1 = 0;
  for (auto& s : ds.samples_) {
    if (CompareUtils::compare(s->benefit_[0], s->benefit_[1]) > 0) {
      s0[countS0++] = (s->benefit_[0] - s->benefit_[1])*(s->benefit_[0] - s->benefit_[1]);
    }
    else {
      s1[countS1++] = (s->benefit_[1] - s->benefit_[0])*(s->benefit_[1] - s->benefit_[0]);
    }
  }
}


long double PairTree::calcSum(const std::vector<long double>& sum, int64_t i,
                              int64_t j) {
  if (i > j) return 0;
  if (i == 0) return sum[j];
  return sum[j] - sum[i - 1];
}


long double PairTree::calcMatchingSums(const std::vector<long double>& a,
  const std::vector<long double>& b) {
  std::vector<long double> sumA(a.size());
  std::vector<long double> sumB(b.size());
  sumA[0] = a[0];
  for (int64_t i = 1; i < a.size(); i++) {
    sumA[i] = sumA[i - 1] + a[i];
  }
  sumB[0] = b[0];
  for (int64_t i = 1; i < b.size(); i++) {
    sumB[i] = sumB[i - 1] + b[i];
  }

  int64_t L = b.size();
  int64_t K = a.size();
  long double matching = 0;
  int64_t right = 0;
  int64_t left = 1;
  for (int64_t i = 0; i <= L - K; i++) {
    if (right < i) {
      right++;
    }
    else {
      left--;
    }
    while (right <= i + K - 1 && b[right] <= a[left]) {
      left++;
      right++;
    }
    matching += std::pow(calcSum(sumB, i, right - 1)
      + calcSum(sumA, left, K - 1), 0.5);
  }

  int64_t left2 = K;
  int64_t right2 = 0;
  for (int64_t i = L - K + 1; i <= L - 1; i++) {
    if (right < i) {
      right++;
    }
    else {
      left--;
    }
    if (right == L) {
      right--;
    }

    while (right <= L - 1 && b[right] <= a[left]) {
      left++;
      right++;
    }
    long double partial1 = calcSum(sumA, left, L - i - 1)
      + calcSum(sumB, i, right - 1);
    left2--;
    while (right2 <= K - (L - i + 1) && b[right2] <= a[left2]) {
      left2++;
      right2++;
    }
    long double partial2 = calcSum(sumA, left2, K - 1)
      + calcSum(sumB, 0, right2 - 1);
    matching += std::pow(partial1 + partial2, 0.5);
  }
  return matching*matching;
}


long double PairTree::calcVarSums(const std::vector<long double>& a,
                                  const std::vector<long double>& b) {
  int K = a.size();
  int L = b.size();
  std::vector<long double> sumBSq(L);
  sumBSq[0] = b[0] * b[0];
  for (int i = 1; i < L; i++) {
    sumBSq[i] = sumBSq[i - 1] + b[i] * b[i];
  }

  long double ans = 0;
  int p = 0;
  for (int i = 0; i < K; i++) {
    while (p < L && a[i] >= b[p]) {
      p++;
    }
    ans += calcSum(sumBSq, 0, p - 1) + (L - p) * a[i] * a[i];
  }
  return ans;
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
