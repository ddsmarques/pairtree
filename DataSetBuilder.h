// This module implements the creation of a dataset based on a CSV file.
// 
// Author: ddsmarques
//

#pragma once
#include "DataSet.h"

class DataSetBuilder {
public:
  DataSetBuilder();

  DataSet buildFromFile(std::string fileName, int64_t classColStart);

  void buildTrainTestFromFile(std::string trainFileName,
                              std::string testFileName,
                              int64_t classColStart,
                              DataSet& trainDS,
                              DataSet& testDS);

private:
  void createAttribute(int col, std::vector<std::vector<std::string>>&& rawFile,
                       DataSet& ds);
  void createClass(int64_t colStart, std::vector<std::vector<std::string>>&& rawFile,
                   DataSet& ds);
  AttributeType getAttributeType(int col, std::vector<std::vector<std::string>>&& rawFile);
  void createSamples(std::vector<std::vector<std::string>>&& rawFile,
                     DataSet& ds, int64_t classColStart);
};
