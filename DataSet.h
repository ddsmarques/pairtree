// This module represents a data set. A data set has attributes and samples.
// 
// Author: ddsmarques
//
#pragma once
#include <list>
#include <vector>
#include <memory>

#include "Attribute.h"
#include "DecisionTreeNode.h"
#include "Sample.h"


class DataSet {
public:
  DataSet();
  template <typename T>
  void addAttribute(std::shared_ptr<Attribute<T>> newAttribute);

  void setClasses(std::vector<std::string>&& classes);

  void addSample(std::shared_ptr<Sample> s);

  AttributeType getAttributeType(int64_t index);

  template <typename T>
  int64_t getValueInx(int64_t attribInx, T value);

  int64_t getClassInx(std::string value);

  int64_t getTotAttributes();

  int64_t getTotClasses();

  int64_t getAttributeSize(int64_t attribInx);

  int64_t getAttributeFrequency(int64_t attribInx, int64_t valueInx);

  std::string getAttributeName(int64_t attribInx);

  std::string getAttributeStringValue(int64_t attribInx, int64_t valueInx);

  std::string getClassValue(int64_t classInx);

  void initAllAttributes(DataSet& ds);

  DataSet getSubDataSet(int64_t attribInx, int64_t valueInx);

  std::pair<int64_t, double> getBestClass();

  double getClassBenefit(int64_t classInx);

  void printTree(std::shared_ptr<DecisionTreeNode> root,
                 std::string fileName);

  std::list<std::shared_ptr<Sample>> samples_;

private:
  void printTreeRec(std::shared_ptr<DecisionTreeNode> node,
                    std::ofstream& ofs, std::string prefix = "");

  std::vector<std::shared_ptr<Attribute<int64_t>>> intAttributes_;
  std::vector<std::shared_ptr<Attribute<double>>> doubleAttributes_;
  std::vector<std::shared_ptr<Attribute<std::string>>> stringAttributes_;
  std::vector<std::pair<AttributeType, int64_t>> attribInfo_;
  std::vector<std::string> classes_;
};
