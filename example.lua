
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
