from setup import *
import contextlib

def make_stochastic(A, orient='col'):    
    if orient == 'row':
        T = A.T
    elif orient == 'col':
        T = A
    else:
        raise Exception("orient must be 'row' or 'col'")
    s = T.sum(axis=0)
    sinks = np.abs(s) < 1e-4
    T[sinks,sinks] = 1
    s[sinks] = 1
    T = T / s
    if orient == 'row':
        T = T.T
    return T


def page_rankify(A, p=0.9, orient='col'):
    T = make_stochastic(A, orient)
    n = len(A)
    r = (1-p)/n
    T = p*T + r
    return T


def eigs(A, orient='col'):
    if orient == 'row':
        B = A.T
    elif orient == 'col':
        B = A
    else:
        raise Exception("orient must be 'row' or 'col'")
    evals, evecs = np.linalg.eig(B)
    s = evecs.sum(axis=0)
    transients = np.abs(s) < 0.01
    s[transients] = 1
    evecs = evecs / s
    if orient == 'row':
        evecs = evecs.T
    return evals, evecs

import time
def render_graphviz(G, node_labels=None, edge_labels=None, w='2in', fmt='svg', show=True): 
    if node_labels is not None:
        for v in G.nodes():
            try:
                l = round(node_labels[v],2)
            except:
                l = node_labels[v]
            G.nodes[v]['label'] = l
        
    if edge_labels is not None:
        for e in G.edges():
            try:
                l = round(edge_labels[e],2)
            except:
                l = edge_labels[e]
            G.edges[e]['label'] = l
    fn = 'im\\graph'+str(round(time.time()*1000))
    dot_file = fn+'.dot'
    im_file = fn+'.'+fmt

    nx.drawing.nx_pydot.write_dot(G,dot_file)
    os.system('dot -T'+fmt+' '+dot_file+' -o '+im_file)
    if show == True:
        display_ims(im_file,w)
    return im_file

def html_call(fn,w='2in'):
    return "<img src='"+fn+"' style='width:"+w+"'>"

def display_ims(fns, w='2in'):
    if isinstance(fns,str):
        fns = [fns]
    html = "</td><td>".join([html_call(fn,w) for fn in fns])
    display(HTML("<table><tr><td>"+html+"</td></tr></table>"))

    
    
def margins(df):
    df = pd.DataFrame(df)
    col_sums = df.sum(axis=0)
    df.loc['TOTAL'] = col_sums
    row_sums = df.sum(axis=1)
    df['TOTAL'] = row_sums
    return df

def get_summary_stats(v):    
    ss = pd.DataFrame(v).describe().T
    ss['SE'] = ss['std'] / np.sqrt(ss['count'])
    return ss

def tile_rows(v,n):
    return np.tile(v,(n,1))

def tile_cols(v,n):
    return np.tile(v[:,np.newaxis],(1,n))

