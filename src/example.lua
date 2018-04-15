-- Copyright (c) 2018 Daniel dos Santos Marques <danielsmarques7@gmail.com>
-- License: BSD 3 clause

name = "trainName"
output = ""

dataset = {
	filename = "..\\..\\..\\..\\datasets\\xor_example.csv",
	classColStart = -1
}

trainMode = {
	trainType = "trainingset"
}
--[[
trainMode = {
	trainType = "testset",
	filename = "bla.csv"
}
trainMode = {
	trainType = "randomsplit",
	ratio = 0.8,
	folds = 3
}
--]]
trees = {}
trees[0] = {
	name = "greedyname",
	treeType = "greedy",
  useNominalBinary = false,
  minSamples = {},
  alphas = {},
  percentiles = 100,
  minLeaf = 30,
  minGain = 0.01,
	height = 2
}
trees[0].alphas[0] = 1
trees[0].minSamples[0] = 1

trees[1] = {
	name = "pairname",
	treeType = "pair",
  useScore = false,
  useNominalBinary = false,
  boundOption = 'DIFF',
  maxBound = 1,
  alphas = {},
  minSamples = {},
  minLeaf = 50,
	height = 10
}
i = 0
for alpha=0.1,1,0.05 do
  trees[1].alphas[i] = alpha
  i = i + 1
end
trees[1].minSamples[0] = 30
trees[1].minSamples[1] = 50
trees[1].minSamples[2] = 100
