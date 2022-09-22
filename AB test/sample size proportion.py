'''
Assume the group to be split 50-50.
After data quality check (remove missing value and outlier), the sample size of the two groups may be not equal. 

'''


def check_sample_size_proportion(n_A, n_B):
    '''
    n_A: int, sample size of control group.
    n_B: int, sample size of test group.
    return: 
    '''
    sum_n = n_A + n_B
    se = ((0.5*(1-0.5))/(sum_n))**0.5
    cl_l, cl_h = 0.5-se*1.96, 0.5+se*1.96
    print('Under 95% CL, the sample size proportion should be in the range of {}% to {}%.'.format(round(cl_l*100,2), round(cl_h*100,2)))
    print('sample size proportion: {}% and {}%.'.format(round(n_A/sum_n*100,2), round(n_B/sum_n*100,2)))
    if cl_l<(n_A/sum_n)<cl_h:
        print('The two groups of samples passed the rationality test of the ratio of the experiment / control group sample.')
        return True
    return False
check_sample_size_proportion(44700,45089)

'''
unequal sample sizes can lead to unequal variances between samples. What's more, unequal sample sizes and variances will affects statistical power and Type I error rates.
refer to https://geoffruddock.com/run-ab-test-with-unequal-sample-size/#:~:text=You%20can%20run%20an%20experiment,to%20achieve%20a%20comparable%20result.
a 10–90 allocation would require 2.8x as many total users to reach a similar outcome as a 50–50 split.
'''
