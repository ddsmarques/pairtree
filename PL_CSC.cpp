#include <iostream>
#include "ReadCSV.h"
#include "DataSetBuilder.h"
#include "PineTree.h"
#include "gurobi_c++.h"
#include "Attribute.h"
#include "GreedyTree.h"
#include "Tester.h"

int main() {
  DataSetBuilder builder;
  //DataSet ds = builder.buildFromFile("C:\\Users\\Daniel\\Documents\\DataSets\\breast-cancer.csv");
  DataSet ds = builder.buildFromFile("C:\\Users\\Daniel\\Documents\\DataSets\\xor_fake.csv");
  //DataSet ds = builder.buildFromFile("C:\\Users\\Daniel\\Documents\\DataSets\\soybean.csv");
  
  PineTree pt;
  std::shared_ptr<ConfigPine> configPine = std::make_shared<ConfigPine>();
  configPine->height = 2;
  configPine->type = ConfigPine::SolverType::INTEGER;
  auto ptNode = pt.createTree(ds, configPine);
  ptNode->printNode();

  GreedyTree gt;
  std::shared_ptr<ConfigGreedy> configGreedy = std::make_shared<ConfigGreedy>();
  configGreedy->height = 2;
  auto gtNode = gt.createTree(ds, configGreedy);
  gtNode->printNode();

  Tester tester;
  tester.test(ptNode, ds);
  tester.test(gtNode, ds);

  system("pause");
  return 0;
}
