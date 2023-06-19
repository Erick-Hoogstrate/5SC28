# Important!

add

def change_reward_function(self, new_reward_function):
        self.reward_fun = new_reward_function

and

def set_discrete_values(self, discrete_size, minmax, div, rnd):
        self.action_space = spaces.Discrete(discrete_size)

        if discrete_size < 3:
                raise ValueError("discrete_size must be at least 3.")
        if discrete_size % 2 == 0:
                raise ValueError("discrete_size must be odd.")

        values = [-minmax]
        initstep = minmax/div  # Calculate the initial step size

        step = initstep
        for i in range(1, int((discrete_size - 1)/2)):
                values.append(round(-step,rnd))
                step = step / div
                
        values.append(0)
        step = step * div
                
        for i in range(1, int((discrete_size - 1)/2)):
                values.append(round(step,rnd))
                step = step * div

        values.append(minmax)

        self.discrete_values = values
        print(f'Discrete value set changed to {values}')

as a function in the UnbalancedDisk.py file
