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

def life_liability_pv(fv,i,mort_tab, defer_yrs=0):
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

#print(life_liability_pv(100,10,[.5,.5]))

# NOTE: With just life_liability_pv() we can calculate how much to charge for a life
#   insurance policy *if* the customer wants to do a lump sum payment instead of
#   an annuity.
# Example
if __name__ == "__main__":
    from neural_net import load_model, NeuralNet
    model=load_model(NeuralNet)
    mort_df=model.get_life_data([[180,'m',72,130,'n','n',3,1,1,'n','n','n',4,'n',0,'n','n',200,'n','n','n','n','n']])
    mort_tab=mort_df[0].to_numpy()
    liability_pv=life_liability_pv(1000000,5,mort_tab)
    print(f'A 1 million dollar life policy for the entered person would cost a ${liability_pv} lump payment up front.')

