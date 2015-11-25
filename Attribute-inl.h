#pragma once
#include "ErrorUtils.h"

template <typename T>
Attribute<T>::Attribute(AttributeType attType)
  : type_(attType), lastInx_(0) {}

template <typename T>
void Attribute<T>::addValue(T value) {
  auto pos = valueInx_.find(value);
  if (pos == valueInx_.end()) {
    valueInx_.insert(std::pair<T, int64_t>(value, lastInx_++));
    inxValue_.push_back(value);
    frequency_.push_back(1);
  }
  else {
    frequency_[pos->second]++;
  }
}

template <typename T>
int64_t Attribute<T>::getInx(T value) {
  auto ans = valueInx_.find(value);
  if (ans == valueInx_.end()) {
    return -1;
  }
  return ans->second;
}

template <typename T>
T Attribute<T>::getValue(int64_t index) {
  ErrorUtils::enforce(index >= 0 && index < inxValue_.size(), "getValue(): Out of bounds");
  return inxValue_[index]; // std::move???
}

template <typename T>
void Attribute<T>::setName(std::string name) {
  name_ = name;
}

template <typename T>
AttributeType Attribute<T>::getType() {
  return type_;
}

template <typename T>
int64_t Attribute<T>::getSize() {
  return valueInx_.size();
}

template <typename T>
int64_t Attribute<T>::getFrequency(int64_t index) {
  ErrorUtils::enforce(index >= 0 && index < frequency_.size(), "getFrequency(): Out of bounds");
  return frequency_[index];
}
