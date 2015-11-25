#include "ReadCSV.h"
#include <fstream>
#include <sstream>

std::vector<std::vector<std::string>> ReadCSV::readFile(std::string fileName) {
  std::vector<std::vector<std::string>> ans;

  std::ifstream file(fileName);
  while (file) {
    auto row = readRow(file);
    ans.push_back(row);
  }
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
