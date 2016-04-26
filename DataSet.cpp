#include "DataSet.h"

#include "CompareUtils.h"

DataSet::DataSet() {}

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

void DataSet::setClasses(std::vector<std::string>&& classes) {
  classes_ = classes;
}

void DataSet::addSample(std::shared_ptr<Sample> s) {
  samples_.push_back(s);
}

void DataSet::eraseSample(std::list<std::shared_ptr<Sample>>::iterator it) {
  samples_.erase(it);
}

AttributeType DataSet::getAttributeType(int64_t index) {
  ErrorUtils::enforce(index >= 0 && index < attribInfo_.size(),
                      "Index out of bounds");
  return attribInfo_[index].first;
}

template <>
int64_t DataSet::getValueInx(int64_t attribInx, int64_t value) {
  ErrorUtils::enforce(attribInx >= 0 && attribInx < attribInfo_.size(),
    "Index out of bounds");
  if (attribInfo_[attribInx].first == AttributeType::INTEGER) {
    return intAttributes_[attribInfo_[attribInx].second]->getInx(value);
  }
  return -1;
}

template <>
int64_t DataSet::getValueInx(int64_t attribInx, double value) {
  ErrorUtils::enforce(attribInx >= 0 && attribInx < attribInfo_.size(),
    "Index out of bounds");
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

int64_t DataSet::getClassInx(std::string value) {
  for (int i = 0; i < classes_.size(); i++) {
    if (classes_[i].compare(value) == 0) {
      return i;
    }
  }
  return -1;
}

int64_t DataSet::getTotAttributes() {
  return attribInfo_.size();
}

int64_t DataSet::getTotClasses() {
  return classes_.size();
}

int64_t DataSet::getAttributeSize(int64_t attribInx) {
  ErrorUtils::enforce(attribInx >= 0 && attribInx < attribInfo_.size(),
                      "Index out of bounds");
  if (attribInfo_[attribInx].first == AttributeType::INTEGER) {
    return intAttributes_[attribInfo_[attribInx].second]->getSize();
  } else if (attribInfo_[attribInx].first == AttributeType::DOUBLE) {
    return doubleAttributes_[attribInfo_[attribInx].second]->getSize();
  } else {
    return stringAttributes_[attribInfo_[attribInx].second]->getSize();
  }
}

int64_t DataSet::getAttributeFrequency(int64_t attribInx, int64_t valueInx) {
  ErrorUtils::enforce(attribInx >= 0 && attribInx < attribInfo_.size(),
                      "Out of bounds");
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

std::string DataSet::getAttributeName(int64_t attribInx) {
  ErrorUtils::enforce(attribInx >= 0 && attribInx < attribInfo_.size(),
    "Out of bounds");
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
  ErrorUtils::enforce(attribInx >= 0 && attribInx < attribInfo_.size(),
    "Out of bounds");
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

std::string DataSet::getClassValue(int64_t classInx) {
  ErrorUtils::enforce(classInx < classes_.size(), "Class index out of bounds");
  return classes_[classInx];
}

void DataSet::initAllAttributes(DataSet& ds) {
  intAttributes_ = ds.intAttributes_;
  doubleAttributes_ = ds.doubleAttributes_;
  stringAttributes_ = ds.stringAttributes_;
  attribInfo_ = ds.attribInfo_;
  classes_ = ds.classes_;
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
    if (CompareUtils::compare(score, bestScore) > 0) {
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
  ErrorUtils::enforce(node != nullptr, "Error: Invalid node in printTreeRec.");
  if (node->getType() == DecisionTreeNode::NodeType::LEAF) {
    ofs << " : " << getClassValue(node->getLeafValue());
  } else if (node->getType() == DecisionTreeNode::NodeType::REGULAR_ORDERED) {
    ofs << std::endl << prefix << getAttributeName(node->getAttribCol())
        << " <= " << getAttributeStringValue(node->getAttribCol(), node->getSeparator());
    printTreeRec(node->getLeftChild(), ofs, prefix + "| ");
    ofs << std::endl << prefix << getAttributeName(node->getAttribCol())
        << " > " << getAttributeStringValue(node->getAttribCol(), node->getSeparator());
    printTreeRec(node->getRightChild(), ofs, prefix + "| ");
  } else {
    for (const auto& child : node->children_) {
      ofs << std::endl << prefix << getAttributeName(node->getAttribCol())
          << " = " << getAttributeStringValue(node->getAttribCol(), child.first);
      printTreeRec(child.second, ofs, prefix + "| ");
    }
  }
}
