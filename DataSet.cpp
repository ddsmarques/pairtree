#include "DataSet.h"

DataSet::DataSet() {

}

void DataSet::addSample(std::shared_ptr<Sample> s) {
  samples_.push_back(s);
}

AttributeType DataSet::getAttributeType(int64_t index) {
  ErrorUtils::enforce(index >= 0 && index < attribInfo_.size(), "Index out of bounds");
  return attribInfo_[index].first;
}

int64_t DataSet::getTotAttributes() {
  return attribInfo_.size();
}

int64_t DataSet::getTotClasses() {
  if (attribInfo_[attribInfo_.size() - 1].first == AttributeType::INTEGER) {
    return intAttributes_[intAttributes_.size() - 1]->getSize();
  } else if (attribInfo_[attribInfo_.size() - 1].first == AttributeType::DOUBLE) {
    return doubleAttributes_[doubleAttributes_.size() - 1]->getSize();
  } else {
    return stringAttributes_[stringAttributes_.size() - 1]->getSize();
  }
}

template <>
void DataSet::addAttribute(std::shared_ptr<Attribute<int64_t>> newAttribute) {
  intAttributes_.push_back(newAttribute);
  attribInfo_.push_back(std::make_pair<AttributeType, int64_t>(AttributeType::INTEGER, intAttributes_.size() - 1));
}

template <>
void DataSet::addAttribute(std::shared_ptr<Attribute<double>> newAttribute) {
  doubleAttributes_.push_back(newAttribute);
  attribInfo_.push_back(std::make_pair<AttributeType, int64_t>(AttributeType::DOUBLE, doubleAttributes_.size() - 1));
}

template <>
void DataSet::addAttribute(std::shared_ptr<Attribute<std::string>> newAttribute) {
  stringAttributes_.push_back(newAttribute);
  attribInfo_.push_back(std::make_pair<AttributeType, int64_t>(AttributeType::STRING, stringAttributes_.size() - 1));
}

template <>
int64_t DataSet::getValueInx(int64_t attribInx, int64_t value) {
  ErrorUtils::enforce(attribInx >= 0 && attribInx < attribInfo_.size(), "Index out of bounds");
  if (attribInfo_[attribInx].first == AttributeType::INTEGER) {
    return intAttributes_[attribInfo_[attribInx].second]->getInx(value);
  }
  return -1;
}

template <>
int64_t DataSet::getValueInx(int64_t attribInx, double value) {
  ErrorUtils::enforce(attribInx >= 0 && attribInx < attribInfo_.size(), "Index out of bounds");
  if (attribInfo_[attribInx].first == AttributeType::DOUBLE) {
    return doubleAttributes_[attribInfo_[attribInx].second]->getInx(value);
  }
  return -1;
}

template <>
int64_t DataSet::getValueInx(int64_t attribInx, std::string value) {
  ErrorUtils::enforce(attribInx >= 0 && attribInx < attribInfo_.size(), "Index out of bounds");
  if (attribInfo_[attribInx].first == AttributeType::STRING) {
    return stringAttributes_[attribInfo_[attribInx].second]->getInx(value);
  }
  return -1;
}

int64_t DataSet::getAttributeSize(int64_t attribInx) {
  ErrorUtils::enforce(attribInx >= 0 && attribInx < attribInfo_.size(), "Index out of bounds");
  if (attribInfo_[attribInx].first == AttributeType::INTEGER) {
    return intAttributes_[attribInfo_[attribInx].second]->getSize();
  } else if (attribInfo_[attribInx].first == AttributeType::DOUBLE) {
    return doubleAttributes_[attribInfo_[attribInx].second]->getSize();
  } else {
    return stringAttributes_[attribInfo_[attribInx].second]->getSize();
  }
}

int64_t DataSet::getAttributeFrequency(int64_t attribInx, int64_t valueInx) {
  ErrorUtils::enforce(attribInx >= 0 && attribInx < attribInfo_.size(), "Out of bounds");
  if (attribInfo_[attribInx].first == AttributeType::INTEGER) {
    return intAttributes_[attribInfo_[attribInx].second]->getFrequency(valueInx);
  }
  else if (attribInfo_[attribInx].first == AttributeType::DOUBLE) {
    return doubleAttributes_[attribInfo_[attribInx].second]->getFrequency(valueInx);
  }
  else {
    return stringAttributes_[attribInfo_[attribInx].second]->getFrequency(valueInx);
  }
}

void DataSet::initAllAttributes(DataSet& ds) {
  intAttributes_ = ds.intAttributes_;
  doubleAttributes_ = ds.doubleAttributes_;
  stringAttributes_ = ds.stringAttributes_;
  attribInfo_ = ds.attribInfo_;
}

DataSet DataSet::getSubDataSet(int64_t attribInx, int64_t valueInx) {
  DataSet newDS;
  newDS.initAllAttributes(*this);
  for (const auto& s : samples_) {
    if (s->inxValue_[attribInx] == valueInx) {
      newDS.addSample(s);
    }
  }
  return newDS;
}

std::pair<int64_t, double> DataSet::getBestClass() {
  int64_t bestInx = 0;
  double bestScore = getClassBenefit(0);
  for (int64_t i = 1; i < getTotClasses(); i++) {
    double score = getClassBenefit(i);
    if (score > bestScore) {
      bestInx = i;
      bestScore = score;
    }
  }
  return std::pair<int64_t, double>(bestInx, bestScore);
}

double DataSet::getClassBenefit(int64_t classInx) {
  double ans = 0;
  for (const auto& s : samples_) {
    ans += s->benefit_[classInx];
  }
  return ans;
}

void DataSet::printTree(std::shared_ptr<DecisionTreeNode> root,
                        std::string fileName) {
  std::ofstream ofs(fileName, std::ofstream::app);
  printTreeRec(root, ofs);
  ofs << std::endl;
  ofs.close();
}

void DataSet::printTreeRec(std::shared_ptr<DecisionTreeNode> node,
                           std::ofstream& ofs, std::string prefix) {
  if (node->getType() == DecisionTreeNode::NodeType::LEAF) {
    ofs << " : " << getAttributeStringValue(attribInfo_.size() - 1,
                                            node->getLeafValue());
  } else {
    for (const auto& child : node->children_) {
      ofs << std::endl << prefix << getAttributeName(node->getAttribCol())
          << " = " << getAttributeStringValue(node->getAttribCol(), child.first);
      printTreeRec(child.second, ofs, prefix + "| ");
    }
  }
}

std::string DataSet::getAttributeName(int64_t attribInx) {
  ErrorUtils::enforce(attribInx >= 0 && attribInx < attribInfo_.size(), "Out of bounds");
  if (attribInfo_[attribInx].first == AttributeType::INTEGER) {
    return intAttributes_[attribInfo_[attribInx].second]->getName();
  }
  else if (attribInfo_[attribInx].first == AttributeType::DOUBLE) {
    return doubleAttributes_[attribInfo_[attribInx].second]->getName();
  }
  else {
    return stringAttributes_[attribInfo_[attribInx].second]->getName();
  }
}

std::string DataSet::getAttributeStringValue(int64_t attribInx, int64_t valueInx) {
  ErrorUtils::enforce(attribInx >= 0 && attribInx < attribInfo_.size(), "Out of bounds");
  if (attribInfo_[attribInx].first == AttributeType::INTEGER) {
    return std::to_string(intAttributes_[attribInfo_[attribInx].second]->getValue(valueInx));
  }
  else if (attribInfo_[attribInx].first == AttributeType::DOUBLE) {
    return std::to_string(doubleAttributes_[attribInfo_[attribInx].second]->getValue(valueInx));
  }
  else {
    return stringAttributes_[attribInfo_[attribInx].second]->getValue(valueInx);
  }
}
