#pragma once
#include "ErrorUtils.h"

#include <algorithm>

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
  return inxValue_[index];
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

template <typename T>
std::string Attribute<T>::getName() {
  return name_;
}

template <typename T>
void Attribute<T>::sortIndexes() {
  std::vector<T> order(valueInx_.size());
  int64_t count = 0;
  for (const auto& it : valueInx_) {
    order[count++] = it.first;
  }
  std::sort(order.begin(), order.end(), &Attribute<T>::lessThan);

  std::map<T, int64_t> newValueInx;
  std::vector<int64_t> newFrequency(order.size());
  for (int64_t i = 0; i < order.size(); i++) {
    newValueInx.insert(std::pair<T, int64_t>(order[i], i));
    newFrequency[i] = frequency_[valueInx_[order[i]]];
  }

  inxValue_ = order;
  valueInx_ = newValueInx;
  frequency_ = newFrequency;
}

template <typename T>
void Attribute<T>::print() {
  std::cout << "Attribute " << name_ << std::endl;
  std::cout << "valueInx_:" << std::endl;
  for (const auto& it : valueInx_) {
    std::cout << it.first << " " << it.second << std::endl;
  }
  std::cout << "frequency_:" << std::endl;
  for (int64_t i = 0; i < frequency_.size(); i++) {
    std::cout << i << " " << frequency_[i] << std::endl;
  }
}
