import os
from scipy.spatial import distance
import cv2
import numpy as np

dataFile = 'inpData.txt'
masterD = dict()

def defineNeighbours( nodeL ):
    numClosest = 4
    neigh = list()
    for ctr in range(len( nodeL ) ):
        refNode = nodeL[ ctr+1 ]
        distL = list()
        for inner in range(len( nodeL ) ):
            #print( ctr, inner )
            u = ( refNode[1]  , refNode[2]   )
            v = ( nodeL[ inner+1 ][1], nodeL[ inner+1 ][2] )
            distL.append( distance.cosine( u, v ) )
        SL = sorted( distL )     
        #print( SL )
        final = [ ( distL.index( x ) + 1 ) for x in SL ] # since we are usin graph conv we need an N x N matrix ..not and N x 4 which te authors have sugested
        #final = [ ( distL.index( x ) + 1 ) for x in SL[ 1:numClosest+1 ] ]# since 1st / 0t ele will always be 0, dist wit itself
        neigh.append( final )
        #neigh[ ctr+1 ] = final
   
    #print( neigh )
    #print( np.asarray( neigh ).shape )
    return neigh    

def orderNodes( nodeL ):  
    xL = list()
    yL = list()
    vall = list()
    nodeOrder = dict()
    store = dict()
    for elem in nodeL:
        xL.append( elem[ 1 ] )
        yL.append( elem[ 2 ] )
        store[ elem[ 1 ]+elem[ 2 ] ] = elem
        vall.append( elem[0] )
    #print( xL )
    #print( yL )
    neoXL = sorted( xL )
    for ctr in range( len(neoXL) ):
        xval = neoXL[ ctr ]
        yval = yL[ xL.index( xval ) ]
        olderarr = store[ xL[ ctr ]+yL[ ctr ] ]
        #print( 'sorted - ', xval, yval, olderarr[5] )
        nodeOrder[ ctr + 1 ] = ( olderarr[0], float( xL[ ctr ] ), float( yL[ ctr ] ), olderarr[3], olderarr[4], olderarr[5] )

    return nodeOrder    

def createLabelInput( nodeL ,totLabels, labLL ):
    retLabs = list()
    for ctr in range( 1, len( nodeL )+1 ):
        init = np.zeros( totLabels ) # init = total labels + 2 slots for REL / IRR
        if nodeL[ctr][-1] != 'IRR':
            init[ labLL.index( nodeL[ctr][-1] ) ] = 1
            init[ -2 ] = 1 ## since the last 2 labels are REL / IRR marking the penultimate one as 
        else:    
            init[-1] = 1# last 2 labels are REL/ IRR so last one is marked 1
        retLabs.append( init )
    #print( retLabs )
    return retLabs

def createOHE( nodeL ):
    #print( nodeL )
    MAX_LEN = 30
    oheList = list()
    special_chars = ',.-+:/%?$£C#()&’'
    for ctr in range( 1, len( nodeL )+1 ):
        ele = nodeL[ ctr ]
        #print( ele )
        maxer = list()
        txt = ele[0].lower()
        for chars in ele[0]:
            charEnc = np.zeros( 52 + 10 + len(special_chars) ) # alphabets + digits + sp chars
            ## first 26 indexes are smal chars and next 26 for large
            if ord( chars ) >= 97 and ord( chars ) <= 122:
                charEnc[ ord( chars )%97 ] = 1
            elif ord( chars ) >= 65 and ord( chars ) <= 90:
                charEnc[ ord( chars )%65 ] = 1
            elif ord( chars ) >= 48 and ord( chars ) <= 57:
                charEnc[ ord( chars )%48 ] = 1
            elif chars in special_chars:
                charEnc[ 61 + special_chars.index( chars ) ] = 1
            
            maxer.append( charEnc )
            if len( maxer ) == MAX_LEN: break

        if len( maxer ) < MAX_LEN: # pad     
            for ctr in range( MAX_LEN - len( maxer ) ):
                charEnc = np.zeros( 52 + 10 + len(special_chars) )
                maxer.append( charEnc )

        oheList.append ( maxer )  ## now the size will be n, 30, 79
    #print( oheList[0] )
    #print( np.asarray( oheList ) )
    return oheList        

def createFeats( fname , nodeL ):
    #img = cv2.imread( 'IMG/'+fname )
    #h,w = img.shape[0], img.shape[1]
    featEle = list()

    for ctr in range( 1, len( nodeL )+1 ):
        ele = nodeL[ ctr ]
        txt = ele[0]
        feats = _get_text_features( txt )
        y1, x1, y2, x2 = float( ele[2] ), float( ele[1] ), ( float( ele[4] ) + float( ele[2] ) ), ( float( ele[3] ) + float( ele[1] ) ) 
        feats.extend( [ y1, x1, y2, x2 ] )
        featEle.append( feats )

    return featEle    

def _get_text_features(data):
    data = str(data)
    arr = data.split(':')
    #data = ' '.join( arr[1:] )
    '''
        Args:
            str, input data
        Returns:
            np.array, shape=(22,);
            an array of the text converted to features
    '''
    # assert type(data) == str, f'Expected type {str}. Received {type(data)}.'
    n_upper = 0
    n_lower = 0
    n_alpha = 0
    n_digits = 0
    n_spaces = 0
    n_numeric = 0
    n_special = 0
    number = 0
    special_chars = {'&': 0, '@': 1, '#': 2, '(': 3, ')': 4, '-': 5, '+': 6,
                     '=': 7, '*': 8, '%': 9, '.': 10, ',': 11, '\\': 12, '/': 13,
                     '|': 14, ':': 15}
    special_chars_arr = np.zeros(shape=len(special_chars))
    # character wise
    for char in data:
        # for lower letters
        if char.islower():
            n_lower += 1
        # for upper letters
        if char.isupper():
            n_upper += 1
        # for white spaces
        if char.isspace():
            n_spaces += 1
        # for alphabetic chars
        if char.isalpha():
            n_alpha += 1
        # for numeric chars
        if char.isnumeric():
            n_numeric += 1
        # array for special chars
        # array for special chars
        n_sp = 0
        if char in special_chars.keys():
            char_idx = special_chars[char]
            # put 1 at index
            special_chars_arr[char_idx] += 1
            n_sp += 1
    ## num of words before first digit
    st = 0
    for char in data:
        if char.isnumeric(): break
        st += 1
    n_b4_first_digit = st
    ## num of words before first digit
    numWords = min( len( data.split() ), 2 )
    # word wise
    for word in data.split():
        # if digit is integer
        try:
            number = int(word)
            n_digits += 1
        except:
            pass
        # if digit is float
        if n_digits == 0:
            try:
                number = float(word)
                n_digits += 1
            except:
                pass
    return ([ n_lower, n_upper, n_spaces, n_sp, n_numeric, n_digits, len(data) ])

def loadData():
    with open( dataFile, 'r' ) as fp:
        ll = fp.readlines()
    
    for elem in ll:
        # f_name	text	xmin	ymin	w	h	label
        arr = elem.strip('\n').split(',')
        if arr[0] in masterD.keys():
            locll = masterD[ arr[0] ]
        else:    
            locll = list()

        locll.append( arr[1:] )
        masterD[ arr[0] ] = locll

    ## iterate once for finding all labels
    labelSet = set()
    for key, item in masterD.items():
        for elem in item: 
            if elem[-1] != 'IRR':
                labelSet.add( elem[-1] )

    totLabels = len( labelSet ) + 2 # the 2 classes are added to indicate REL / IRR
    labLL = list( labelSet )
    print( labLL )
    batcMaster = dict()

    for key, item in masterD.items():
        orderedNodes = orderNodes( item )
        neighBours = defineNeighbours( orderedNodes )
        labels = createLabelInput( orderedNodes, totLabels, labLL )
        OHE    = createOHE( orderedNodes )
        features = createFeats( key, orderedNodes )
        batcMaster[ key ] = ( orderedNodes, neighBours, labels, OHE , features )

    return batcMaster, totLabels

#loadData()
