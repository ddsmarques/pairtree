// This module represents a data set. A data set has attributes and samples.
// 
// Author: ddsmarques
//
#pragma once
#include <list>
#include <vector>

#include "Attribute.h"
#include "Sample.h"


class DataSet {
public:
  DataSet();
  template <typename T>
  void addAttribute(Attribute<T>& newAttribute);

  void addSample(Sample& s);

  AttributeType getAttributeType(int64_t index);

  template <typename T>
  int64_t getValueInx(int64_t attribInx, T value);

  int64_t getTotAttributes();

  int64_t getTotClasses();

  int64_t getAttributeSize(int64_t attribInx);

  int64_t getAttributeFrequency(int64_t attribInx, int64_t valueInx);

  std::list<Sample> samples_;

private:
  std::vector<Attribute<int64_t>> intAttributes_;
  std::vector<Attribute<double>> doubleAttributes_;
  std::vector<Attribute<std::string>> stringAttributes_;
  std::vector<std::pair<AttributeType, int64_t>> attribInfo_;
};
