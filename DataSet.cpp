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
  for (auto s : samples_) {
    if (s->inxValue_[attribInx] == valueInx) {
      newDS.addSample(s);
    }
  }
  return newDS;
}
