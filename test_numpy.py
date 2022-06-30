import numpy as np
import datetime

def gen_nums(deterministic=False):
    for i in range(10):
        if deterministic:
            np.random.seed(0+i)
        else:
            np.random.seed()
        location = np.random.uniform(-1.1,1.1,3)
        location[2] = np.random.uniform(0.25,1.00)
        
        rotation = np.array([np.random.uniform(-0.4, 0.4),np.random.uniform(-0.4, 0.4),np.random.uniform(0, np.pi)])
        print(location, rotation)

if __name__ == '__main__':
    print('deterministic')
    gen_nums(deterministic=True)
    print('nondeterministic')
    gen_nums(deterministic=False)
    

    print('deterministic')
    gen_nums(deterministic=True)
    print('nondeterministic')
    gen_nums(deterministic=False)
