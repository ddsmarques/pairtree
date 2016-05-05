#include "DataSetBuilder.h"

#include "Converter.h"
#include "ErrorUtils.h"
#include "Logger.h"
#include "ReadCSV.h"

#include <utility>
#include <iostream>
#include <set>

DataSetBuilder::DataSetBuilder() {
}

DataSet DataSetBuilder::buildFromFile(std::string fileName,
                                      int64_t classColStart) {
  DataSet ds;
  ReadCSV csvReader;
  auto rawFile = csvReader.readFile(fileName);
  ErrorUtils::enforce(rawFile.size() > 0, "Empty file");

  Logger::log() << "Started building file " << fileName;
  int64_t totAttrib = 0;
  if (classColStart <= 0) {
    totAttrib = rawFile[0].size() - 1;
  } else {
    totAttrib = classColStart;
  }

  // Add all attributes to the dataset
  for (int i = 0; i < totAttrib; i++) {
    createAttribute(i, std::move(rawFile), ds);
  }
  // Add all classes to the dataset
  createClass(classColStart, std::move(rawFile), ds);

  // Create all samples
  for (int i = 1; i < rawFile.size(); i++) if (rawFile[i].size() > 0) {
    std::shared_ptr<Sample> s = std::make_shared<Sample>(ds.getTotAttributes(), ds.getTotClasses());
    // Sample attributes
    for (int j = 0; j < totAttrib; j++) {
      if (ds.getAttributeType(j) == AttributeType::INTEGER) {
        s->inxValue_[j] = ds.getValueInx(j, Converter::fromString<int64_t>(rawFile[i][j]));
      } else if (ds.getAttributeType(j) == AttributeType::DOUBLE) {
        s->inxValue_[j] = ds.getValueInx(j, Converter::fromString<double>(rawFile[i][j]));
      } else {
        s->inxValue_[j] = ds.getValueInx(j, rawFile[i][j]);
      }
    }
    // Sample class benefit
    if (classColStart <= 0) {
      s->benefit_[ds.getClassInx(rawFile[i][totAttrib])] = 1;
    } else {
      for (int j = classColStart; j < rawFile[i].size(); j++) {
        s->benefit_[j - classColStart] = -Converter::fromString<double>(rawFile[i][j]);
      }
    }
    ds.addSample(s);
  }
  Logger::log() << "Finished building file " << fileName;
  return ds;
}

void DataSetBuilder::createAttribute(int col, std::vector<std::vector<std::string>>&& rawFile, DataSet& ds) {
  auto colType = getAttributeType(col, std::forward<std::vector<std::vector<std::string>>&&>(rawFile));
  if (colType == AttributeType::INTEGER) {
    std::shared_ptr<Attribute<int64_t>> attrib = std::make_shared<Attribute<int64_t>>(AttributeType::INTEGER);

    attrib->setName(rawFile[0][col]);
    for (int i = 1; i < rawFile.size(); i++) if (col < rawFile[i].size()) {
      attrib->addValue(Converter::fromString<int64_t>(rawFile[i][col]));
    }
    attrib->sortIndexes();
    ds.addAttribute<int64_t>(attrib);
  } else if (colType == AttributeType::DOUBLE) {
    std::shared_ptr<Attribute<double>> attrib = std::make_shared<Attribute<double>>(AttributeType::DOUBLE);

    attrib->setName(rawFile[0][col]);
    for (int i = 1; i < rawFile.size(); i++) if (col < rawFile[i].size()) {
      attrib->addValue(Converter::fromString<double>(rawFile[i][col]));
    }
    attrib->sortIndexes();
    ds.addAttribute<double>(attrib);
  } else {
    std::shared_ptr<Attribute<std::string>> attrib = std::make_shared<Attribute<std::string>>(AttributeType::STRING);

    attrib->setName(rawFile[0][col]);
    for (int i = 1; i < rawFile.size(); i++) if (col < rawFile[i].size()) {
      attrib->addValue(rawFile[i][col]);
    }
    attrib->sortIndexes();
    ds.addAttribute<std::string>(attrib);
  }
}

void DataSetBuilder::createClass(int64_t colStart,
                                 std::vector<std::vector<std::string>>&& rawFile,
                                 DataSet& ds) {
  if (colStart <= 0) {
    std::set<std::string> values;
    for (int64_t i = 1; i < rawFile.size(); i++) if (rawFile[i].size() > 0) {
      values.insert(rawFile[i][rawFile[i].size() - 1]);
    }
    std::vector<std::string> classes;
    for (auto&& s : values) {
      classes.push_back(s);
    }
    std::sort(classes.begin(), classes.end());
    ds.setClasses(std::move(classes));
  } else {
    ds.setClasses(std::vector<std::string>(rawFile[0].begin() + colStart,
                  rawFile[0].end()));
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
