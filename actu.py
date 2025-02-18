from neural_net import load_model, NeuralNet
def payout_pv(fv, n, i):
    '''Calculates the present value of a payment in n years at a given interest rate
    Args:
        fv: The final payment amount
        n: Number of years til payout
        i: Interest/Yield rate enter as a percentage (e.g., for 5% enter 5)
    Returns:
        Present value of payment
    '''
    return fv*((1+i/100)**-n)

#print(payout_pv(100,7,10))

def annuity_pmt(n,i,pv):
    '''Calculates the payment for a annuity with a given duration, yield, and
    present value
    Args:
        n: years duration
        i: Interest/Yield rate enter as a percentage (e.g., for 5% enter 5)
    pv: Present value
    Returns:
        Payment amount
    '''
    i=i/100
    return pv/((1-(1+i)**-n)/i)

#print(annuity_pmt(10,5,1000))

def annuity_pv(n,i,pmt):
    '''Calculates the present value of a annuity
    Args:
        n: periods
        i: yield per period
        pmt: Payment
    Returns:
        annuity_pv
    '''
    i=i/100
    return (1-(1+i)**-n)/i

#print(annuity_pv(5,5,1))

def life_liability_pv_mu(fv,i,mort_tab, defer_yrs=0):
    '''Calculates the expected pv of life insurance policy liability
    Args:
        fv: payment amount on death
        i: Interest/Yield rate enter as a percentage (e.g., for 5% enter 5)
        mort_table: The odds of them dying in each year
        defer_yrs: Number of years until policy start (e.g., if they are 20,
          but we know they won't die until 25 at the soonest, then defer_yrs=4)
    Returns:
        expected pv of life insurance policy liability
    '''
    pv=0
    for n,mort in enumerate(mort_tab,1+defer_yrs):
        pv+=payout_pv(fv,n,i)*mort
    return pv

def life_liability_pv_q(fv,i,mort_tab, defer_yrs=0,quart=.5):
    '''Calculates the percentile pv of life insurance policy liability
    Args:
        fv: payment amount on death
        i: Interest/Yield rate enter as a percentage (e.g., for 5% enter 5)
        mort_table: The odds of them dying in each year
        defer_yrs: Number of years until policy start (e.g., if they are 20,
          but we know they won't die until 25 at the soonest, then defer_yrs=4)
        quart: number between 0 and 1
    Returns:
        expected pv of life insurance policy liability
    '''
    pv=0
    q=0
    for n,mort in enumerate(mort_tab,1+defer_yrs):
        q+=mort
        if q>=quart:
            return payout_pv(fv,n,i)

#print(life_liability_pv(100,10,[.5,.5]))

def life_pmt_mu(fv,i,mort_tab,defer_yrs=0):
    '''Calculates a fixed payment annuity payment
    to match the liability
    Args:
        fv: payment amount on death
        i: Interest/Yield rate enter as a percentage (e.g., for 5% enter 5)
        mort_table: The odds of them dying in each year
        defer_yrs: Number of years until policy start (e.g., if they are 20,
          but we know they won't die until 25 at the soonest, then defer_yrs=4)
        quart: number between 0 and 1
    Returns:
        The expected payment amount to equal liability
    '''
    liability_pv=life_liability_pv_mu(fv,i,mort_tab,defer_yrs)
    simple_annuity_pv_mu=0
    for n,mort in enumerate(mort_tab,1+defer_yrs):
        simple_annuity_pv_mu+=annuity_pv(n,i,1)*mort
    pmt=liability_pv/simple_annuity_pv_mu
    return pmt

# All of the life functions can be reused for a life insurance policy
# that ends at a certain age by just removing the last however many rows from the table
# The new table won't add up to 1, but that's because then there is a chance people won't die in that period


def actu_str(inputs,fv,age):
    '''Returns a string with information about insurance for an individual
    Args:
        inputs: the paramaters for the neural net prediction
        fv: How much you want the policy to payout
        lia_dif: How many years til the liability begins (i.e., how many years til you turn 25)
    Returns:
        Str
    '''
    I=1
    def_years=0
    if age<25:
        def_years=25-age
    path='models/'+str(int(age+def_years))+'.pth'
    model=load_model(path)
    mort_df=model.get_life_data([inputs])
    # Currently this is working on the unsmoothed data, add smoothing later
    mort_tab=mort_df[0].to_numpy()
    liability_pv=life_liability_pv_mu(fv,I,mort_tab,def_years)
    liability_pv_med=life_liability_pv_q(fv,I,mort_tab,def_years)
    fixed_pmt=life_pmt_mu(fv,I,mort_tab,def_years)
    cost_str=f'A \${fv:,.2f} life policy for this {age} year-old person would cost a \${liability_pv:.2f} lump payment up front.\n'+\
    f'The same policy has median liability present value of \${liability_pv_med:.2f}\n'+\
    f'This policy could be payed for by a lifetime fixed annuity of \${fixed_pmt:.2f} per year.'
    return cost_str

# Example
if __name__ == "__main__":
    print(actu_str([180,'m',72,130,'n','n',3,1,1,'n','n','n',4,'n',0,'n','n',200,'n','n','n','n','n'],250000,20))