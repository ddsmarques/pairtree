// Copyright (c) 2018 Daniel dos Santos Marques <danielsmarques7@gmail.com>
// License: BSD 3 clause

#include "ReadCSV.h"

#include "Logger.h"

#include <fstream>
#include <sstream>

std::vector<std::vector<std::string>> ReadCSV::readFile(std::string fileName) {
  Logger::log() << "Started reading file " << fileName;
  std::vector<std::vector<std::string>> ans;

  std::ifstream file(fileName);
  while (file) {
    auto row = readRow(file);
    if (row.size() > 0) {
      ans.push_back(row);
    }
  }
  Logger::log() << "Finished reading file " << fileName;
  return ans;
}

std::vector<std::string> ReadCSV::readRow(std::istream& str) {
  std::vector<std::string> ans;
  std::string line;
  std::getline(str, line);
  std::stringstream lineStream(line);
  std::string cell;
  while (std::getline(lineStream, cell, ','))
  {
    ans.push_back(cell);
  }
  return ans;
}
