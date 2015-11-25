// This module implements the creation of a dataset based on a CSV file.
// 
// Author: ddsmarques
//

#pragma once
#include "DataSet.h"

class DataSetBuilder {
public:
  DataSetBuilder();

  DataSet buildFromFile(std::string fileName);

private:
  void createAttribute(int col, std::vector<std::vector<std::string>>&& rawFile, DataSet& ds);
  AttributeType getAttributeType(int col, std::vector<std::vector<std::string>>&& rawFile);
};
