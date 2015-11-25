#include "DataSet.h"

DataSet::DataSet() {

}

void DataSet::addSample(Sample& s) {
  samples_.push_back(s);
}

AttributeType DataSet::getAttributeType(int64_t index) {
  //ENFORCE(index < attribInfo_.size(), "Index out of bounds");
  return attribInfo_[index].first;
}

int64_t DataSet::getTotAttributes() {
  return attribInfo_.size();
}

int64_t DataSet::getTotClasses() {
  if (attribInfo_[attribInfo_.size() - 1].first == AttributeType::INTEGER) {
    return intAttributes_[intAttributes_.size() - 1].getSize();
  } else if (attribInfo_[attribInfo_.size() - 1].first == AttributeType::DOUBLE) {
    return doubleAttributes_[doubleAttributes_.size() - 1].getSize();
  } else {
    return stringAttributes_[stringAttributes_.size() - 1].getSize();
  }
}

template <>
void DataSet::addAttribute(Attribute<int64_t>& newAttribute) {
  intAttributes_.push_back(newAttribute);
  attribInfo_.push_back(std::make_pair<AttributeType, int64_t>(AttributeType::INTEGER, intAttributes_.size() - 1));
}

template <>
void DataSet::addAttribute(Attribute<double>& newAttribute) {
  doubleAttributes_.push_back(newAttribute);
  attribInfo_.push_back(std::make_pair<AttributeType, int64_t>(AttributeType::DOUBLE, doubleAttributes_.size() - 1));
}

template <>
void DataSet::addAttribute(Attribute<std::string>& newAttribute) {
  stringAttributes_.push_back(newAttribute);
  attribInfo_.push_back(std::make_pair<AttributeType, int64_t>(AttributeType::STRING, stringAttributes_.size() - 1));
}

template <>
int64_t DataSet::getValueInx(int64_t attribInx, int64_t value) {
  //ENFORCE(attribInx < attribInfo_.size(), "Index out of bounds");
  if (attribInfo_[attribInx].first == AttributeType::INTEGER) {
    return intAttributes_[attribInfo_[attribInx].second].getInx(value);
  }
  return -1;
}

template <>
int64_t DataSet::getValueInx(int64_t attribInx, double value) {
  //ENFORCE(attribInx < attribInfo_.size(), "Index out of bounds");
  if (attribInfo_[attribInx].first == AttributeType::DOUBLE) {
    return doubleAttributes_[attribInfo_[attribInx].second].getInx(value);
  }
  return -1;
}

template <>
int64_t DataSet::getValueInx(int64_t attribInx, std::string value) {
  //ENFORCE(attribInx < attribInfo_.size(), "Index out of bounds");
  if (attribInfo_[attribInx].first == AttributeType::STRING) {
    return stringAttributes_[attribInfo_[attribInx].second].getInx(value);
  }
  return -1;
}

int64_t DataSet::getAttributeSize(int64_t attribInx) {
  //ENFORCE(attribInx < attribInfo_.size(), "Index out of bounds");
  if (attribInfo_[attribInx].first == AttributeType::INTEGER) {
    return intAttributes_[attribInfo_[attribInx].second].getSize();
  } else if (attribInfo_[attribInx].first == AttributeType::DOUBLE) {
    return doubleAttributes_[attribInfo_[attribInx].second].getSize();
  } else {
    return stringAttributes_[attribInfo_[attribInx].second].getSize();
  }
}

int64_t DataSet::getAttributeFrequency(int64_t attribInx, int64_t valueInx) {
  //ENFORCE(attribInx < attribInfo_.size(), "Out of bounds");
  if (attribInfo_[attribInx].first == AttributeType::INTEGER) {
    return intAttributes_[attribInfo_[attribInx].second].getFrequency(valueInx);
  }
  else if (attribInfo_[attribInx].first == AttributeType::DOUBLE) {
    return doubleAttributes_[attribInfo_[attribInx].second].getFrequency(valueInx);
  }
  else {
    return stringAttributes_[attribInfo_[attribInx].second].getFrequency(valueInx);
  }
}

