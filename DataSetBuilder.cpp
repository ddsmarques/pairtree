#include "DataSetBuilder.h"
#include "ReadCSV.h"
#include "Converter.h"
#include "ErrorUtils.h"

#include <utility>
#include <iostream>

DataSetBuilder::DataSetBuilder() {
}

DataSet DataSetBuilder::buildFromFile(std::string fileName) {
  DataSet ds;
  ReadCSV csvReader;
  auto rawFile = csvReader.readFile(fileName);
  ErrorUtils::enforce(rawFile.size() > 0, "Empty file");

  for (int i = 0; i < rawFile[0].size(); i++) {
    createAttribute(i, std::move(rawFile), ds);
  }
  for (int i = 1; i < rawFile.size(); i++) if (rawFile[i].size() > 0) {
    Sample s(ds.getTotAttributes(), ds.getTotClasses());
    for (int j = 0; j < rawFile[i].size(); j++) {
      if (ds.getAttributeType(j) == AttributeType::INTEGER) {
        s.inxValue_[j] = ds.getValueInx(j, Converter::fromString<int64_t>(rawFile[i][j]));
      } else if (ds.getAttributeType(j) == AttributeType::DOUBLE) {
        s.inxValue_[j] = ds.getValueInx(j, Converter::fromString<double>(rawFile[i][j]));
      } else {
        s.inxValue_[j] = ds.getValueInx(j, rawFile[i][j]);
      }
    }
    //std::cout << rawFile[i][rawFile[i].size() - 1] << std::endl;
    s.benefit_[s.inxValue_[ds.getTotAttributes() - 1]] = 1;
    ds.addSample(s);
  }
  return ds;
}

void DataSetBuilder::createAttribute(int col, std::vector<std::vector<std::string>>&& rawFile, DataSet& ds) {
  auto colType = getAttributeType(col, std::forward<std::vector<std::vector<std::string>>&&>(rawFile));
  //auto colType = getAttributeType(col, rawFile);
  if (colType == AttributeType::INTEGER) {
    Attribute<int64_t> attrib(AttributeType::INTEGER);

    attrib.setName(rawFile[0][col]);
    for (int i = 1; i < rawFile.size(); i++) if (col < rawFile[i].size()) {
      attrib.addValue(Converter::fromString<int64_t>(rawFile[i][col]));
    }
    ds.addAttribute<int64_t>(attrib);
  } else if (colType == AttributeType::DOUBLE) {
    Attribute<double> attrib(AttributeType::DOUBLE);

    attrib.setName(rawFile[0][col]);
    for (int i = 1; i < rawFile.size(); i++) if (col < rawFile[i].size()) {
      attrib.addValue(Converter::fromString<double>(rawFile[i][col]));
    }
    ds.addAttribute<double>(attrib);
  } else {
    Attribute<std::string> attrib(AttributeType::STRING);

    attrib.setName(rawFile[0][col]);
    for (int i = 1; i < rawFile.size(); i++) if (col < rawFile[i].size()) {
      attrib.addValue(rawFile[i][col]);
    }
    ds.addAttribute<std::string>(attrib);
  }
}

AttributeType DataSetBuilder::getAttributeType(int col, std::vector<std::vector<std::string>>&& rawFile) {
  bool isInteger = true;
  bool isDouble = true;
  for (int i = 1; i < rawFile.size(); i++) {
    if (col < rawFile[i].size()) {
      isInteger = isInteger & Converter::isInteger(rawFile[i][col]);
      isDouble = isDouble & Converter::isDouble(rawFile[i][col]);
    }
  }

  if (!isInteger && !isDouble) {
    return AttributeType::STRING;
  } else if (!isInteger) {
    return AttributeType::DOUBLE;
  } else if (isInteger) {
    return AttributeType::INTEGER;
  }
}
