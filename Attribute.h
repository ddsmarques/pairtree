// This module represents an attribute which can assume one of 3 types: INTEGER, DOUBLE and STRING.
// 
// Author: ddsmarques
//

#pragma once
#include <vector>
#include <map>
#include <string>

enum class AttributeType {INTEGER, DOUBLE, STRING};

template <typename T>
class Attribute {
public:
	Attribute(AttributeType attType);

	void addValue(T value);
  int64_t getInx(T value);
	T getValue(int64_t index);
  void setName(std::string name);
  AttributeType getType();
  int64_t getSize();
  int64_t getFrequency(int64_t index);
	
private:
  const AttributeType type_;
  std::map<T, int64_t> valueInx_;
  std::vector<T> inxValue_;
  std::vector<int64_t> frequency_;
  int64_t lastInx_;
  std::string name_;
};

#include "Attribute-inl.h"