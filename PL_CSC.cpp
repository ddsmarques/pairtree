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
  DataSet ds = builder.buildFromFile("C:\\Users\\Daniel\\Documents\\DataSets\\breast-cancer.csv");
  //DataSet ds = builder.buildFromFile("C:\\Users\\Daniel\\Documents\\DataSets\\xor_fake.csv");
  
  PineTree pt;
  auto ptNode = pt.createTree(ds, 2, PineTree::SolverType::INTEGER);
  ptNode->printNode();

  GreedyTree gt;
  auto gtNode = gt.createTree(ds, 2);
  gtNode->printNode();

  Tester tester;
  tester.test(ptNode, ds);
  tester.test(gtNode, ds);

  system("pause");
  return 0;
}
