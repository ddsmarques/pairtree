// This module loads a CSV into memory.
// 
// Author: ddsmarques
//
#pragma once
#include <string>
#include <vector>

class ReadCSV {
public:
  std::vector<std::vector<std::string>> readFile(std::string fileName);

private:
  std::vector<std::string> readRow(std::istream& str);
};
