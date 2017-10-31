import math

def readMesh(fileLoc):
	''' Takes in the file location of the FEpX mesh file. It then goes through and reads
		and parses the data to only return the relevant Surface Nodes as a list with each
		surface seperated into their own respected surface subsection. If they are located
		on multiple surfaces than the nodes go to the multiple surface subsection.
	'''
	nodes=[]
	surfaceNodes=[]
	with open(fileLoc) as f:
		data = f.readlines()
    	
	for line in data:
		words=line.split()
#		print(words)
		lenWords=len(words)
		if not words:
			continue
		if lenWords==4:
			nums=wordParser(words)
			#print(nums)
			lenNums=len(nums)
			nodes.append(nums)
	
	surfaceNodes=nodeSurface(nodes)
	return surfaceNodes
	
def wordParser(listVals):
	'''
	Read in the string list and parse it into a floating list
	'''
	numList=[]
	for str in listVals:
		num=float(str)
		numList.append(num)
		#print(num)
	return numList
	
def nodeSurface(nodes):
	'''
	Read in all the nodes and then determines which ones are on a surface. It returns a
	set of surface nodes
	'''
	surfaceNodes=[]
	for nod in nodes:
		loc=nod[1:]
		if 0.0 in loc:
			surfaceNodes.append(nod)
		elif 1.0 in loc:
			surfaceNodes.append(nod)
	return surfaceNodes
	
def createBCSFile(sNodes,vel,bcsCond,fileName,fileLoc):
	'''
	Read in surfaceNodes, strain rate/vel, and boundary bounded condition.
	It outputs a file in the following format
	#Nodes
		Node	T/F		T/F		T/F		velX velY velZ
		...
	The file name is specified by the user without the .bcs and so is the location
	to save it at.
	'''
	outputList=[]
	dirct=fileLoc+fileName+'.bcs'
	for nod in sNodes:
		output=outputStruct(nod,vel,bcsCond)
		outputList.append(output)
	writeOutput(outputList,dirct)
	
def outputStruct(nod,vel,bcsCond):
	'''
	Takes in node, velocity, and boundary condition setting.
	Creates an output list from the above
	'''
	isGrip=False
	output=[0]*7
	if bcsCond=='grip':
		isGrip=True
	output[0]=int(nod[0])
	loc=nod[1:]
	if isGrip:
		output[1]='F'
		output[2]='F'
		output[4]=0.000000000
		output[5]=0.000000000
	else:
		for i in range(0,2):
			if loc[i]==0:
				output[i+1]='T'
			else:
				output[i+1]='F'
			output[i+4]=0.000000000
	
	if loc[2]==0 or loc[2]==1:
		output[3]='T'
		if loc[2]==1:
			output[6]=vel
		else:
			output[6]=0.000000000
	else:
		output[3]='F'
		output[6]=0.000000000
	return output

def writeOutput(outputList,dirct):
	'''
	Writes the actual .bcs file in the directory file provided
	'''
	lenOutput=len(outputList)
	maxVal=outputList[lenOutput-1]
	logMax=math.floor(math.log10(maxVal[0]))
	with open(dirct,'w') as f:
		f.write(str(lenOutput)+'\n')
		for list in outputList:
			s=''
			for i in range(0,7):
				if i<4:
					if i==0:
						val=list[i]
						
						if val==0:
							val=1
								
						logCur=math.floor(math.log10(val))
						s+=' '*(logMax-logCur)
					s+='     '+str(list[i])
				elif i==4:
					s+='     '+'{0:.6e}'.format(list[i])
				else:
					s+=' '+'{0:.6e}'.format(list[i])
			print(s)
			f.write(s+'\n')
	
		

fileLoc='/Users/robertcarson/Research_Local_Code/fepx-devl/Examples/ControlMode/n2.mesh'
vel=0.005;
bcsCondition='symmetric'
fileName='example'
fileLocation='/Users/robertcarson/Research_Local_Code/fepx/Jobs/'
surfaceNodes=readMesh(fileLoc)
#createBCSFile(surfaceNodes,vel,bcsCondition,fileName,fileLocation)
