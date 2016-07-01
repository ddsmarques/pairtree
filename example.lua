
name = "trainName"
output = "C:\\Users\\Daniel\\Documents\\tmp\\"

dataset = {
	filename = "C:\\\\Users\\Daniel\\Documents\\DataSets\\xor_fake.csv",
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
	name = "pinename",
	treeType = "pine",
	height = 2,
	solver = "CONTINUOUS"
}

trees[1] = {
	name = "greedyname",
	treeType = "greedy",
	height = 2
}
trees[2] = {
	name = "pairname",
	treeType = "pair",
  useScore = false,
  maxBound = 1,
  alphas = {},
  minSamples = {},
  minLeaf = 50,
	height = 10
}
i = 0
for alpha=0.1,1,0.05 do
  trees[2].alphas[i] = alpha
  i = i + 1
end
trees[2].minSamples[0] = 30
trees[2].minSamples[1] = 50
trees[2].minSamples[2] = 100
