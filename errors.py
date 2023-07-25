import numpy as np
import plotly.graph_objects as go

def getErrors(bpmES, bpmGT, timesES, timesGT, metrics):
    """ Computes various error/quality measures"""
    if type(bpmES) == list:
        bpmES = np.expand_dims(bpmES, axis=0)
    if type(bpmES) == np.ndarray:
        if len(bpmES.shape) == 1:
            bpmES = np.expand_dims(bpmES, axis=0)

    err = []
    
    for m in metrics:
        if m == 'RMSE':
            e = RMSEerror(bpmES, bpmGT, timesES, timesGT)
        elif m == 'MAE':
            e = MAEerror(bpmES, bpmGT, timesES, timesGT)
        elif m == 'MAPE':
            e = MAPEerror(bpmES, bpmGT, timesES, timesGT)
        elif m == 'MAX':
            e = MAXError(bpmES, bpmGT, timesES, timesGT)
        elif m == 'PCC':
            e = PearsonCorr(bpmES, bpmGT, timesES, timesGT)
        elif m == 'CCC':
            e = LinCorr(bpmES, bpmGT, timesES, timesGT)
       elif m == 'dPCC':
            e = [bpmES, bpmGT]
        elif m == 'dCCC':
            e = [bpmES, bpmGT]

        err.append(e)

    return err


def RMSEerror(bpmES, bpmGT, timesES=None, timesGT=None):
    """ Computes RMSE """

    diff = bpm_diff(bpmES, bpmGT, timesES, timesGT)
    n, m = diff.shape  # n = num channels, m = bpm length
    df = np.zeros(n)
    for j in range(m):
        for c in range(n):
            df[c] += np.power(diff[c, j], 2)

    # -- final RMSE
    RMSE = float(np.sqrt(df/m))
    return RMSE


def MAEerror(bpmES, bpmGT, timesES=None, timesGT=None):
    """ Computes MAE """

    diff = bpm_diff(bpmES, bpmGT, timesES, timesGT)
    n, m = diff.shape  # n = num channels, m = bpm length
    df = np.sum(np.abs(diff), axis=1)

    # -- final MAE
    MAE = df/m
    return MAE

def MAPEerror(bpmES, bpmGT, timesES=None, timesGT=None):
    """ Computes MAE """

    diff = bpm_diff(bpmES, bpmGT, timesES, timesGT, normalize=True)
    n, m = diff.shape  # n = num channels, m = bpm length
    df = np.sum(np.abs(diff), axis=1)

    # -- final MAE
    MAPE = (df/m) * 100
    return MAPE


def MAXError(bpmES, bpmGT, timesES=None, timesGT=None):
    """ computes MAX """

    diff = bpm_diff(bpmES, bpmGT, timesES, timesGT)
    n, m = diff.shape  # n = num channels, m = bpm length
    df = np.max(np.abs(diff), axis=1)

    # -- final MAX
    MAX = df
    return MAX


def PearsonCorr(bpmES, bpmGT, timesES=None, timesGT=None):
    """ Computes PCC """
    from scipy import stats

    diff = bpm_diff(bpmES, bpmGT, timesES, timesGT)
    n, m = diff.shape  # n = num channels, m = bpm length

    if m < 2:
        print('> Warning: Correlation cannot be calculated for signals with len < 2. Returning NaN')
        return np.nan

    CC = np.zeros(n)
    for c in range(n):
        # -- corr
        r, p = stats.pearsonr(diff[c, :]+bpmES[c, :], bpmES[c, :])
        CC[c] = r
    return CC


def LinCorr(bpmES, bpmGT, timesES=None, timesGT=None):
    """ Computes CCC """
    diff = bpm_diff(bpmES, bpmGT, timesES, timesGT)
    n, m = diff.shape  # n = num channels, m = bpm length

    if m < 2:
        print('> Warning: Correlation cannot be calculated for signals with len < 2. Returning NaN')
        return np.nan

    CCC = np.zeros(n)
    for c in range(n):
        # -- Lin's Concordance Correlation Coefficient
        ccc = concordance_correlation_coefficient(bpmES[c, :], diff[c, :]+bpmES[c, :])
        CCC[c] = ccc
    return CCC


def printErrors(RMSE, MAE, MAX, PCC, CCC):
    print("\n    * Errors: RMSE = %.2f, MAE = %.2f, MAX = %.2f, PCC = %.2f, CCC = %.2f" %
          (RMSE, MAE, MAX, PCC, CCC))


def displayErrors(bpmES, bpmGT, timesES=None, timesGT=None):
    """"Plots errors"""
    if type(bpmES) == list:
        bpmES = np.expand_dims(bpmES, axis=0)
    if type(bpmES) == np.ndarray:
        if len(bpmES.shape) == 1:
            bpmES = np.expand_dims(bpmES, axis=0)


    if (timesES is None) or (timesGT is None):
        timesES = np.arange(m)
        timesGT = timesES

    diff = bpm_diff(bpmES, bpmGT, timesES, timesGT)
    n, m = diff.shape  # n = num channels, m = bpm length
    df = np.abs(diff)
    dfMean = np.around(np.mean(df, axis=1), 1)

    # -- plot errors
    fig = go.Figure()
    name = 'Ch 1 (µ = ' + str(dfMean[0]) + ' )'
    fig.add_trace(go.Scatter(
        x=timesES, y=df[0, :], name=name, mode='lines+markers'))
    if n > 1:
        name = 'Ch 2 (µ = ' + str(dfMean[1]) + ' )'
        fig.add_trace(go.Scatter(
            x=timesES, y=df[1, :], name=name, mode='lines+markers'))
        name = 'Ch 3 (µ = ' + str(dfMean[2]) + ' )'
        fig.add_trace(go.Scatter(
            x=timesES, y=df[2, :], name=name, mode='lines+markers'))
    fig.update_layout(xaxis_title='Times (sec)',
                      yaxis_title='MAE', showlegend=True)
    fig.show()

    # -- plot bpm Gt and ES
    fig = go.Figure()
    GTmean = np.around(np.mean(bpmGT), 1)
    name = 'GT (µ = ' + str(GTmean) + ' )'
    fig.add_trace(go.Scatter(x=timesGT, y=bpmGT,
                             name=name, mode='lines+markers'))
    ESmean = np.around(np.mean(bpmES[0, :]), 1)
    name = 'ES1 (µ = ' + str(ESmean) + ' )'
    fig.add_trace(go.Scatter(
        x=timesES, y=bpmES[0, :], name=name, mode='lines+markers'))
    if n > 1:
        ESmean = np.around(np.mean(bpmES[1, :]), 1)
        name = 'ES2 (µ = ' + str(ESmean) + ' )'
        fig.add_trace(go.Scatter(
            x=timesES, y=bpmES[1, :], name=name, mode='lines+markers'))
        ESmean = np.around(np.mean(bpmES[2, :]), 1)
        name = 'E3 (µ = ' + str(ESmean) + ' )'
        fig.add_trace(go.Scatter(
            x=timesES, y=bpmES[2, :], name=name, mode='lines+markers'))

    fig.update_layout(xaxis_title='Times (sec)',
                      yaxis_title='BPM', showlegend=True)
    fig.show()


def bpm_diff(bpmES, bpmGT, timesES=None, timesGT=None, normalize=False):
    n, m = bpmES.shape  # n = num channels, m = bpm length

    if (timesES is None) or (timesGT is None):
        timesES = np.arange(m)
        timesGT = timesES

    diff = np.zeros((n, m))
    for j in range(m):
        t = timesES[j]
        i = np.argmin(np.abs(t-timesGT))
        for c in range(n):
            if not normalize:
                diff[c, j] = bpmGT[i]-bpmES[c, j]
            else:
                diff[c, j] = (bpmGT[i]-bpmES[c, j]) / bpmGT[i]
    return diff

def concordance_correlation_coefficient(bpm_true, bpm_pred):
    cor=np.corrcoef(bpm_true, bpm_pred)[0][1]
    mean_true = np.mean(bpm_true)
    mean_pred = np.mean(bpm_pred)
    
    var_true = np.var(bpm_true)
    var_pred = np.var(bpm_pred)
    
    sd_true = np.std(bpm_true)
    sd_pred = np.std(bpm_pred)
    
    numerator = 2*cor*sd_true*sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred)**2

    return numerator/denominator


