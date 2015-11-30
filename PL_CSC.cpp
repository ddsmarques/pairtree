#include <iostream>
#include "ReadCSV.h"
#include "DataSetBuilder.h"
#include "PineTree.h"
#include "gurobi_c++.h"
#include "Attribute.h"
#include "GreedyTree.h"
#include "Tester.h"

int main() {
  ReadCSV reader;
  auto contents = reader.readFile("C:\\Users\\Daniel\\Downloads\\contact-lenses.csv");
  for (int i = 0; i < contents.size(); i++) {
    for (int j = 0; j < contents[i].size(); j++) {
      std::cout << contents[i][j] << " ";
    }
    std::cout << std::endl;
  }
  DataSetBuilder builder;
  //DataSet ds = builder.buildFromFile("C:\\Users\\Daniel\\Downloads\\contact-lenses.csv");
  DataSet ds = builder.buildFromFile("C:\\Users\\Daniel\\Downloads\\xor.csv");
  for (auto s : ds.samples_) {
    for (auto i : s->inxValue_) {
      std::cout << i << " ";
    }
    std::cout << std::endl;
  }
  
  PineTree pt;
  auto ptNode = pt.createBackBone(ds, 2, PineTree::SolverType::INTEGER);
  ptNode->printNode();

  GreedyTree gt;
  auto gtNode = gt.createBackBone(ds, 3);
  gtNode->printNode();

  PineTree ptTree;
  auto ptTreeNode = ptTree.createTree(ds, 2, PineTree::SolverType::INTEGER);
  ptTreeNode->printNode();
  
  Tester tester;
  tester.test(ptNode, ds);
  tester.test(gtNode, ds);
  tester.test(ptTreeNode, ds);
  system("pause");
  return 0;
}
