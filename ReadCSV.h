// Copyright (c) 2018 Daniel dos Santos Marques <danielsmarques7@gmail.com>
// License: BSD 3 clause
//
// This module loads a CSV into memory.
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
