-- Local Modular

-- Synchronize local G
Gloc1 = sync(G1,G2,G3,G4,G5)

-- Synchronize local K
Kloc1 = sync(Gloc1,E1)

-- Create local supervisors
Sloc1 = supc(Gloc1, Kloc1)

print("----------")
print(infom(Kloc1))
print(infom(Sloc1))

Kloc1 = minimize(Kloc1)

Sloc1 = minimize(Sloc1)

print("----------")
print(infom(Kloc1))
print(infom(Sloc1))

-- Add to Nadzoru
export( Sloc1 )

