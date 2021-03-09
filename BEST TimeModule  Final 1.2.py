import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from tabulate import tabulate

TimeInstances=1
TS=2
TimeInterval=5

SizePopulation=16
Generations=100
CrossoverProbability=0.5
MutationProbability=0.1

RouteMat=np.array([[3,1,2,0],[3,1,0,2],[1,3,2,0],[1,3,0,2]])
DemandMat=np.array([[0,2,4,2,0,3,2,0],[1,0,3,2,2,0,1,1],[4,1,0,0,2,5,0,0],[3,1,0,0,1,0,0,0]])
BusTravelTimeMat=np.array([[0,4,2,1,0,3,1,1],[2,0,3,2,1,0,2,1],[3,6,0,0,2,5,0,0],[5,1,0,0,4,1,0,0]])

CostperMin=8
TicketPrice=10
FixedCost=5

print("Initial Data")
print("\nThe routes followed by the buses are given as:")
print(tabulate(RouteMat, headers="", tablefmt="grid"))
print("\nThe demand at each node is given as: ")
print(tabulate(DemandMat, headers="", tablefmt="grid"))
print("\nThe time taken for a bus to go from each node is: ")
print(tabulate(BusTravelTimeMat, headers="", tablefmt="grid"))

N_Routes=RouteMat.shape[0] #Number of rows of Route Matrix gives number of routes in the network
N_Nodes=RouteMat.shape[1] #Number of columns of Route Matrix gives number of nodes in the network

Parameters=5
MinCounter=0
RouteCombiTime=0
PassengerUpCount=0
NoShowMember=0
GlobalPassengerUpCount=np.zeros((SizePopulation,N_Nodes))
LocalPassengerUpCount=np.zeros((1,N_Nodes))
PassengerDownCount=np.zeros((1,N_Nodes))
Plot_Fitness=np.zeros((Generations,SizePopulation+1))

RouteMatSeq=np.zeros((N_Routes,N_Nodes))
for i in range(N_Routes):
    for r in range(N_Nodes):
        for j in range(N_Nodes):
            if RouteMat[i][j]==r+1:
                RouteMatSeq[i][r]=j+1
print("\nThe route sequence of the bus deployed is given as:")
print(tabulate(RouteMatSeq, headers="", tablefmt="grid"))

Demand_TS=np.zeros((1,TS))
for i in range(TS):
    for j in range(N_Nodes):
        for k in range(N_Nodes):
            Demand_TS[0][i]=Demand_TS[0][i]+DemandMat[k][N_Nodes*i+j]
TotalDemand=np.sum(Demand_TS)
print("\nTotal demand in each time slot is given as:")
print(tabulate(Demand_TS, headers="", tablefmt="grid"))

#OP_RouteCombiMat=np.array([[0,0,0,0],[0,0,0,1],[0,0,1,0],[0,0,1,1],[0,1,0,0],[0,1,0,1],[0,1,1,0],[0,1,1,1],[1,0,0,0],[1,0,0,1],[1,0,1,0],[1,0,1,1],[1,1,0,0],[1,1,0,1],[1,1,1,0],[1,1,1,1]])
OP_RouteCombiMat=np.random.randint(2,size=(SizePopulation,N_Routes))
print("\nThe Initial route combination is given as: ")
print(tabulate(OP_RouteCombiMat, headers="", tablefmt="grid"))

print("\nGENETIC ALGORITHM INITIALIZATION")
#Setting up the initial data
def FitnessFunction(a,b,c):
    z=round((((1.0/(1.0+a))+((b+1.0)/(c+1.0)))/4)**5, 4)
    return(z)
    #Where a=Unsatisfied Passengers, b=Revenue, c=Cost to BEST, z=Fitness Function

#Step1: ********************************** GENERATION **********************************************

for t in range(TimeInstances):
    print("\nTime Instance=",t)
    for g in range(Generations):
        print("\nGeneration=",g+1)
        Plot_Fitness[g][0]=g+1
        ParametersMat = np.zeros((SizePopulation, Parameters))
        print("\nInitial Generation")
        for m in range(SizePopulation):
            for im in range(N_Routes):
                if OP_RouteCombiMat[m][im]==1:
                   for LR5 in range(N_Nodes):
                       if RouteMatSeq[im][LR5]!=0:
                          if LR5<N_Nodes-1: 
                             SourceAddress=RouteMatSeq[im][LR5]
                             DestinationAddress=RouteMatSeq[im][LR5+1]
                             if DestinationAddress!=0:
                                for LR6 in range(LR5+1,N_Nodes-1): 
                                    LocalPassengerUpCount[0][LR5]=LocalPassengerUpCount[0][LR5]+DemandMat[int(SourceAddress)-1][int(RouteMatSeq[im][LR6])+int(N_Nodes*np.floor(MinCounter/TimeInterval))-1]
                                GlobalPassengerUpCount[m][im]=GlobalPassengerUpCount[m][im]+np.sum(LocalPassengerUpCount)
                                LocalPassengerUpCount=np.zeros((1,N_Nodes))
                                MinCounter=MinCounter+BusTravelTimeMat[int(SourceAddress)-1][int(DestinationAddress)-1]+N_Nodes*np.floor(MinCounter/TimeInterval)
                   RouteCombiTime=RouteCombiTime+MinCounter
                else:
                   NoShowMember=NoShowMember+1
                MinCounter=0
            ParametersMat[m][0]=RouteCombiTime #Total Time kithna laga for the Combination
            RouteCombiTime=0
            for LR7 in range(N_Routes):
                ParametersMat[m][1]=ParametersMat[m][1]+GlobalPassengerUpCount[m][LR7] # Satisfied Passengers
            #ParametersMat[m][2]=ParametersMat[m][0]*CostperMin+FixedCost*NoShowMember # Cost to BEST
            ParametersMat[m][2]=ParametersMat[m][0]*CostperMin+FixedCost*N_Routes #Cost To BEST
            NoShowMember=0    
            ParametersMat[m][3]=ParametersMat[m][1]*TicketPrice # Revenue
            ParametersMat[m][4]=TotalDemand-ParametersMat[m][1]#Unsatified Passengers
        GlobalPassengerUpCount=np.zeros((SizePopulation,N_Nodes))
        print("\nParametersMat")
        print(tabulate(ParametersMat, headers="", tablefmt="grid"))
        
#Step2: ********************************REPRODUCTION************************************************
                        
        #Calculation of fitness of the DNA Strands (Member Combination)
        print("\nReproduction")
        Sum=0
        Const1=0
        Sol=np.zeros((SizePopulation,3))
        for LR8 in range(SizePopulation):
            print("LR8")
            print(LR8)
            print("a")
            print(ParametersMat[LR8][4])
            print("b")
            print(ParametersMat[LR8][3])
            print("c")
            print(ParametersMat[LR8][2])
            print("Before Sol",Sol[LR8][0])
            Sol[LR8][0]=FitnessFunction(ParametersMat[LR8][4],ParametersMat[LR8][3],ParametersMat[LR8][2]) #Calculating the member fitness
            #print("Plot_Fitness", Plot_Fitness)
            Plot_Fitness[Generations-1][LR8+1]=Sol[LR8][0]
            print("After Sol",Sol[LR8][0])
            print("Before Sum",Sum)
            Sum=Sum+Sol[LR8][0]
            print("After Sum",Sum)
            print("SOL Before Dividing with sum")
            print(tabulate(Sol, headers="", tablefmt="grid"))

        for LR9 in range(SizePopulation):
            Sol[LR9][1]=Sol[LR9][0]/Sum
            if i>0: 
               Sol[LR9][2]=Sol[LR9][1]+Sol[LR9-1][2]
            else:
               Sol[0][2]=Sol[0][1]
            Present=np.concatenate((OP_RouteCombiMat,Sol),axis=1)
            
        print("\nPresent")
        print(tabulate(Present, headers="", tablefmt="grid"))
        

        Next=np.zeros((SizePopulation,1))
        Decider=np.zeros((SizePopulation,1))
        for i in range(SizePopulation):
            Decider[i][0]=np.random.random()
        print("\nDecider")
        print(tabulate(Decider, headers="", tablefmt="grid"))
        for LR10 in range(SizePopulation):
            for LR11 in range(SizePopulation):
                if Decider[LR10][0]>0 and Decider[LR10][0]<Sol[0][2]: 
                   Next[LR10][0]=0
                if Decider[LR10][0]>=Sol[LR11][2] and Decider[LR10][0]<=Sol[LR11+1][2]:
                   Next[LR10][0]=LR11+2
        ParametersMat=np.zeros((SizePopulation,Parameters))
        print("\nNext")
        print(tabulate(Next, headers="", tablefmt="grid"))

        New_Population=np.zeros((SizePopulation,N_Routes))
        for LR12 in range(SizePopulation):
            for LR13 in range(SizePopulation):
                if Next[LR12][0]==LR13+1:
                   for LR14 in range(N_Routes):
                       New_Population[LR12][LR14]=OP_RouteCombiMat[LR13][LR14]
        print("\nNew Population Crossover")
        print(tabulate(New_Population, headers="", tablefmt="grid"))
        
#Step3:***************************************************CROSSOVER****************************************************
        print("\nCROSSOVER")
        Const2=0
        CrossoverPoint=2
        print("\nNew Population Before Crossover")
        print(tabulate(New_Population, headers="", tablefmt="grid"))
        for LR15 in range(SizePopulation//2):
            Crossover=np.random.random()#rand()
            print("\nCrossoverProbability=",Crossover)
            if Crossover>CrossoverProbability:
               for LR16 in range(CrossoverPoint,N_Routes):
                   Const2=New_Population[LR15][LR16]
                   New_Population[LR15][LR16]=New_Population[LR15+(SizePopulation//2)][LR16]
                   New_Population[LR15+(SizePopulation//2)][LR16]=Const2
            print("\nPopulation After Crossover")
            print(tabulate(New_Population, headers="", tablefmt="grid"))
        for LR20 in range(SizePopulation):
            for LR21 in range(N_Routes):
                OP_RouteCombiMat[LR20][LR21]=New_Population[LR20][LR21]
        print("\nOP_After Crossover")
        print(tabulate(OP_RouteCombiMat, headers="", tablefmt="grid"))

#Step4 : #############################################Mutation#########################################################
        print("\nMutation")
        Mutation = np.random.random()  #Mutation=0.09
        
        print("\nMutation=",Mutation)
        if Mutation<MutationProbability:
           RandomMember=np.random.randint(SizePopulation,size=(1,1))
           print(RandomMember)
           print("\nOP_RouteCombiMatMemeber Before Mutation=", OP_RouteCombiMat[RandomMember][:])

           for LR22 in range(N_Routes):
               
               print(OP_RouteCombiMat[int(RandomMember)][LR22])
               if OP_RouteCombiMat[int(RandomMember)][LR22]==0:
                   
                   OP_RouteCombiMat[int(RandomMember)][LR22]=1
               elif OP_RouteCombiMat[int(RandomMember)][LR22]==1:
                   
                   OP_RouteCombiMat[int(RandomMember)][LR22]=0
               print("\nOP_RouteCombiMatMember after Mutation=", OP_RouteCombiMat[RandomMember][:])

    if t==0:
       Solution=New_Population
    if t>0: 
       Solution=np.concatenate((Solution,New_Population),axis=0)
print("\nFinal Solution")
print(tabulate(Solution, headers="", tablefmt="grid"))

#print(tabulate(Plot_Fitness, headers="", tablefmt="grid"))
X=np.zeros((Generations,1))
Y=np.zeros((Generations,1))
for t in range(SizePopulation):
    for i in range(Generations):
        X[i][0]=Plot_Fitness[i][0]
        Y[i][0]=Plot_Fitness[i][t+1]
        
    plt.plot(X,Y)

plt.title("Generations vs Objective Function")
plt.xlabel("Generations")
plt.ylabel("Objective Function")        
plt.show()

