from neural_net import load_model, NeuralNet
from utils import policy_type_format

def get_mort_tab(age,inputs,smooth=100):
    '''
    Retrieves mortality table data for a specified age 
    using a trained model. 
    If the age is less than 25, the function adjusts to use the model for 
    age 25. It loads the appropriate model, retrieves life 
    data with optional smoothing, and returns it as a numpy array.

    Parameters:
        age (int): The age for which mortality data is required.
        If age is less than 25, the function defaults to 25.
        inputs (list): A list of inputs required by the model
        to generate predictions. These inputs must match
        the model's expected input format.
        smooth (int, optional): The percentage of the output 
        to apply Gaussian smoothing. Defaults to 100 for
        fully smoothed outputs.

    Returns:
        numpy.ndarray: A numpy array of the mortality 
        table data derived from the model's predictions.

    Usage:
        >>> age = 30
        >>> inputs = [180, 'm', 72, 130, 'n', 'n', 3, 1, 1, 
        'n', 'n', 'n', 4, 'n', 0, 'n', 'n', 200, 'n', 'n', 'n', 'n', 'n']
        >>> mortality_table = get_mort_tab(age, inputs)
        >>> print(mortality_table)

    Note:
        - The function ensures that the model used for predictions 
        is at least for age 25, reflecting minimum policy age limits.
        - The `sigma` value for Gaussian smoothing is set to
          7.5, which influences the smoothness of the output.
    '''
    if age<25:
        def_years=25-age
    else:
        def_years=0
    path='models/'+str(int(age+def_years))+'.pth'
    model=load_model(path)
    mort_df=model.get_life_data([inputs],False,smooth,sigma=7.5)
    mort_tab=mort_df[0].to_numpy()
    return mort_tab

def payout_pv(fv, n, i):
    '''Calculates the present value of a payment 
    in n years at a given interest rate
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

def simple_annuity_fv(n,i,pmt):
    '''
    Calculates the future value of a simple annuity given the number 
    of periods, the interest rate, and the periodic payment. 
    The function first computes the present value of a unit
      annuity using another function `annuity_pv`, then scales
        this by the periodic payment and compounds it to the end 
        of the period specified.

    Parameters:
        n (int): The number of periods over which the annuity is paid.
        i (float): The interest rate per period, expressed 
        as a percentage. For example, 5 for 5%.
        pmt (float): The amount of each periodic payment.

    Returns:
        float: The future value of the annuity at the end of `n` periods.

    Usage:
        >>> future_value = simple_annuity_fv(10, 5, 100)
        >>> print(future_value)
        # This will print the future value of an annuity where 
        # $100 is paid each period for 10 periods at an interest rate of 5%.

    Note:
        - The interest rate `i` is expected as a percentage 
        and is converted within the function to a decimal by dividing by 100.
        - This function depends on `annuity_pv`, 
        which must be defined elsewhere in your codebase. 
        This function calculates the present value of an annuity 
        of $1 per period for `n` periods at a given interest rate.
    '''
    pv=annuity_pv(n,i,1)
    i=i/100
    return pv*((1+i)**n)

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

def duration_liability_mu(fv,i,mort_tab,defer_yrs,duration):
    '''
    Calculates the modified present value of a liability
    that is expected to be paid over a specified duration, 
    taking into account mortality rates and the time value of
    money. The liability's payments are adjusted by the 
    probability of survival from a mortality table and are 
    deferred by a certain number of years.

    Parameters:
        fv (float): The future value or the payment 
        amount of the liability at each period.
        i (float): The annual discount rate as a decimal. 
        For example, 0.05 for 5%.
        mort_tab (numpy.ndarray): A table of mortality 
        probabilities corresponding to each year of life. 
        This table is used to adjust the cash flows 
        based on the likelihood of survival to each payment period.
        defer_yrs (int): The number of years before the 
        liability payments begin.
        duration (int): The total number of years from the 
        beginning of the period to the end of the liability payments.

    Returns:
        float: The present value of the liability, 
        adjusted for mortality and the time value of money.

    Usage:
        >>> fv = 1000
        >>> i = 0.03
        >>> mort_tab = numpy.array([0.99, 0.98, 0.97, 0.96])  # Example mortality probabilities
        >>> defer_yrs = 2
        >>> duration = 6
        >>> pv = duration_liability_mu(fv, i, mort_tab, defer_yrs, duration)
        >>> print(pv)
        # Outputs the present value of the liability considering the given parameters.

    Note:
        - It is crucial that the `mort_tab` array is at least 
        as long as the `duration` minus `defer_yrs`. If it is not, the function will encounter an index out of range error.
        - This function assumes that `life_liability_pv_mu`,
        a function to calculate the present value of 
        life contingent liabilities, is defined elsewhere 
        in your codebase and is used here to perform the 
        actual present value calculation.
    '''
    mort_tab=mort_tab[:int(duration-defer_yrs)]
    return life_liability_pv_mu(fv,i,mort_tab,defer_yrs)

def duration_pmt_mu(fv,i,mort_tab,defer_yrs,duration):
    '''
    Calculates the periodic payment that needs to 
    be made to match the present value of a future liability,
    adjusted for mortality, over a specified duration and 
    deferment period. The function computes the present value of 
    the liability first and then determines the equivalent 
    annuity payment that matches this present value using
    mortality-adjusted annuity factors.

    Parameters:
        fv (float): The future value or the payment amount of 
        each liability payment.
        i (float): The discount rate per period as a decimal.
        For example, 0.05 for 5%.
        mort_tab (numpy.ndarray): A table of mortality 
        probabilities corresponding to each year of life. 
        This table is used to adjust the annuity payments 
        based on the probability of survival to each payment period.
        defer_yrs (int): The number of years before 
        the annuity payments begin.
        duration (int): The total number of years from
        the beginning of the period to the end of the annuity payments.

    Returns:
        float: The periodic payment amount for an annuity that has the same present value as the calculated liability, adjusted for mortality.

    Usage:
        >>> fv = 100000
        >>> i = 0.04
        >>> mort_tab = numpy.array([0.99, 0.98, 0.97, 0.96])  # Example mortality probabilities
        >>> defer_yrs = 1
        >>> duration = 5
        >>> periodic_payment = duration_pmt_mu(fv, i, mort_tab, defer_yrs, duration)
        >>> print(periodic_payment)
        # Outputs the required periodic annuity payment that equates to the present value of the liability.

    Note:
        - The mortality table (`mort_tab`) must be long enough to cover the `duration` minus `defer_yrs`; otherwise, the function might attempt to access beyond the end of the array, leading to an error.
        - The function internally calls `duration_liability_mu` to compute the liability's present value and `annuity_pv` to calculate the present value of an annuity of $1 per period, adjusted for mortality.
        - The `total_p` in the loop accumulates the probability of survival, which influences the calculation of the final annuity present value.
    '''
    liability_pv=duration_liability_mu(fv,i,mort_tab,defer_yrs,duration)
    simple_annuity_pv_mu=0
    total_p=0
    mort_tab=mort_tab[:int(duration-defer_yrs)]
    for n,mort in enumerate(mort_tab,1+defer_yrs):
        simple_annuity_pv_mu+=annuity_pv(n,i,1)*mort
    simple_annuity_pv_mu+=annuity_pv(duration,i,1)*(1-total_p)
    return liability_pv/simple_annuity_pv_mu

def annual_to_monthly_pmt(annual_payment,i):
    '''
    Converts an annual payment amount to its equivalent monthly payment amount, accounting for the compounding interest rate. The function calculates the effective monthly interest rate from the annual rate and uses it to compute the monthly payment that has the same present value as the given annual payment.

    Parameters:
        annual_payment (float): The total amount paid annually.
        i (float): The annual interest rate as a percentage (e.g., 5 for 5%).

    Returns:
        float: The equivalent monthly payment amount.

    Usage:
        >>> annual_payment = 1200  # Equivalent to $1200 paid once a year
        >>> annual_interest_rate = 5  # Annual interest rate of 5%
        >>> monthly_payment = annual_to_monthly_pmt(annual_payment, annual_interest_rate)
        >>> print(f'The equivalent monthly payment is: ${monthly_payment:.2f}')
        # This will print the equivalent monthly payment amount given the annual payment and interest rate.

    Note:
        - The calculation of the monthly interest rate involves converting the annual rate to a monthly compound rate, which is then used to determine the equivalent monthly payments using the future value of a simple annuity.
        - This function assumes that `simple_annuity_fv` is a function defined elsewhere in your codebase that calculates the future value of a simple annuity given the number of periods, interest rate, and periodic payment.
    '''
    monthly_i=((1+i/100)**(1/12)-1)*100
    return annual_payment/simple_annuity_fv(12,monthly_i,1)

def years_left_mu(mort_tab,def_yrs):
    '''
    Calculates the expected number of years left, adjusted for mortality probabilities, and adds deferment years. This is often used in actuarial science to determine the expected remaining lifetime or duration of payments considering the likelihood of survival each year.

    Parameters:
        mort_tab (numpy.ndarray or list): An array or list of mortality probabilities for each subsequent year. These probabilities should sum to 1 or represent the probability distribution of survival for each year.
        def_yrs (int): The number of years to defer or add to the calculated expected years. This could represent a delay before an annuity starts or any other kind of deferment in financial calculations.

    Returns:
        float: The adjusted mean lifetime, calculated as the sum of each year's probability weighted by the year number, plus the deferment years.

    Usage:
        >>> mort_tab = [0.99, 0.98, 0.95, 0.90, 0.80]  # Example mortality probabilities
        >>> def_yrs = 5  # Deferment of 5 years
        >>> expected_years = years_left_mu(mort_tab, def_yrs)
        >>> print(f'Expected years left, adjusted for mortality and deferment: {expected_years}')
        # This will calculate and print the expected number of years left, considering mortality probabilities and deferment.

    Note:
        - The function assumes that the mortality table provides a probability of survival for each year starting from the first year listed. If the probabilities do not sum to 1, the calculation still proceeds under the assumption that they represent the distribution of survival probabilities accurately.
        - The index `i` in the loop starts at 0, which implies that `i` represents the year number starting from 0. Adjust accordingly if your year numbering starts from 1.
    '''
    mu=0
    for i,p in enumerate(mort_tab):
        mu+=i*p
    return mu+def_yrs

# All of the life functions can be reused for a life insurance policy
# that ends at a certain age by just removing the last however many rows from the table
# The new table won't add up to 1, but that's because then there is a chance people won't die in that period


def actu_str(inputs,fv,age,policy_type='fl',duration=None,payment_type=None,I=1):
    '''Returns a string with information about insurance for an individual
    Args:
        inputs: the paramaters for the neural net prediction
        fv: How much you want the policy to payout
        lia_dif: How many years til the liability begins (i.e., how many years til you turn 25)
    Returns:
        Str
    '''
    def_years=0
    if age<25:
        def_years=25-age
    path='models/'+str(int(age+def_years))+'.pth'
    model=load_model(path)
    mort_df=model.get_life_data([inputs],False,True,sigma=10)
    mort_tab=mort_df[0].to_numpy()
    match payment_type:
        case 'Lump':
            if policy_type=='fl':
                liability_pv=life_liability_pv_mu(fv,I,mort_tab,def_years)
            else:
                liability_pv=duration_liability_mu(fv,I,mort_tab,def_years,duration)
            return f'A \${fv:,.2f} life {policy_type_format(policy_type,duration).lower()} policy for this {age} year-old person would cost a \${liability_pv:.2f} lump payment up front.'
        case 'Annual':
            if policy_type=='fl':
                pmt=life_pmt_mu(fv,I,mort_tab,def_years)
            elif policy_type=='fd':
                pmt=duration_pmt_mu(fv,I,mort_tab,def_years,duration)
            else:
                pmt=duration_pmt_mu(fv,I,mort_tab,def_years,1)
            return f'A \${fv:,.2f} life {policy_type_format(policy_type,duration).lower()} policy for this {age} year-old person would cost an annual payment of \${pmt:.2f}.'
        case 'Monthly':
            if policy_type=='fl':
                pmt=life_pmt_mu(fv,I,mort_tab,def_years)
            elif policy_type=='fd':
                pmt=duration_pmt_mu(fv,I,mort_tab,def_years,duration)
            else:
                pmt=duration_pmt_mu(fv,I,mort_tab,def_years,1)
            pmt=annual_to_monthly_pmt(pmt,I)
            return f'A \${fv:,.2f} life {policy_type_format(policy_type,duration).lower()} policy for this {age} year-old person would cost a monthly payment of \${pmt:.2f}.'
        case _:
            if policy_type=='fl':
                liability_pv=life_liability_pv_mu(fv,I,mort_tab,def_years)
            else:
                liability_pv=duration_liability_mu(fv,I,mort_tab,def_years,duration)
            #liability_pv_med=life_liability_pv_q(fv,I,mort_tab,def_years)
            if policy_type=='fl':
                pmt=life_pmt_mu(fv,I,mort_tab,def_years)
            elif policy_type=='fd':
                pmt=duration_pmt_mu(fv,I,mort_tab,def_years,duration)
            monthly_pmt=annual_to_monthly_pmt(pmt,I)
            cost_str=f'A \${fv:,.2f} life {policy_type_format(policy_type,duration).lower()} policy for this {age} year-old person would cost a \${liability_pv:.2f} lump payment up front.\n'+\
            f'This policy could be payed for by a lifetime an annuity of \${pmt:.2f} per year or a monthly payment of \${monthly_pmt:.2f}'
            return cost_str

# Example
if __name__ == "__main__":
    print(actu_str([160,'m',72,130,'n','n',3,1,1,'n','n','n',4,'n',0,'n','n',200,'n','n','y','y','y'],250000,20))