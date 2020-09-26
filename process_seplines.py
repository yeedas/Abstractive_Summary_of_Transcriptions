
import os, sys, glob
import pandas as pd

def gen_sep_lines(filename):
    fname = open(filename, 'r')
    flines = fname.readlines()
    count = 0
    prev_line = ""
    speaker_name = ""
    prev_speaker_name = ""
    final_lines = []
    sent_count = 0
    prev_sent_count = 0
    valid_line = True
    #end_of_transcript = False
    for line in flines:
        #if end_of_transcript == True:
            #break

        if line.find("____________________") != -1:
            count = count + 1

            if count == 1:
                print ("Found start of line")
                continue
            else:
                #end_of_transcript = True
                print ("Found end of line")
                break
        
        if count == 0:
            continue

        # Skip the noisy sentences
        if line.find("joined the conference") != -1:
            continue
        if line.find("left the conference") != -1:
            continue

        # Skip the time stamp
        elecount = 0
        for ele in line:
            elecount = elecount + 1
            if ele == '<':
                continue
            if ele == '>':
                # Skip the space after >
                elecount = elecount + 1
                line = line[elecount:]
                break
       
        # Parse the rest of the line which has speaker name at the start of line
        elecount = 0
        for ele in line:
            elecount = elecount + 1
            # Get the name of the speaker
            if ele == ':':
                valid_line = True
                speaker_name = line[:(elecount-1)]

                # Skip the space after :
                line = line[elecount+1:]
                #print ("actual line: ", line)
                lines = line.split('\n')
                for l in lines:
                    #print ("l is :", l)
                    line = line+l

                line = speaker_name+" said "+line
                break

        if valid_line == True:
            line = line.split('\n')[0]
            #line = line.strip('\n')
            if line[-1] != '.':
                line = line[:-1]+'.'
            final_lines.append(line)

    print (final_lines)

    df = pd.DataFrame(final_lines)
    #df.columns = ["Transcript Sentences"]
    #df.to_csv('results_seplines.csv', header=True)
    df.to_csv('results_seplines.txt', header=None, index=None, sep='\n', mode='w')

    return "results_seplines.txt"
