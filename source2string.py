#!/usr/bin/env python3

# Convert a supplied file into a C string
# Usage: source2string -i infile outfile 
# Based on code given in https://stackoverflow.com/questions/410980

import sys
import re
import ntpath

def is_printable_ascii(byte):
    return byte >= ord(' ') and byte <= ord('~')

def needs_escaping(byte):
    return byte == ord('\"') or byte == ord('\\')

def stringify_nibble(nibble):
    if nibble < 10:
        return chr(nibble + ord('0'))
    return chr(nibble - 10 + ord('a'))

def write_byte(of, byte):
    if is_printable_ascii(byte):
        if needs_escaping(byte):
            of.write('\\')
        of.write(chr(byte))
    elif byte == ord('\n'):
        of.write('\\n"\n"')
    else:
        of.write('\\x')
        of.write(stringify_nibble(byte >> 4))
        of.write(stringify_nibble(byte & 0xf))

def mk_valid_identifier(s):
    s = re.sub('^[^_a-z]', '_', s)
    s = re.sub('[^_a-z0-9]', '_', s)
    return s

def main():
    # `xxd -i` compatibility
    if len(sys.argv) != 4 or sys.argv[1] != "-i":
        print("Usage: source2string -i infile outfile")
        exit(2)

    # Use just the filename for the 
    with open(sys.argv[2], "rb") as infile:
        with open(sys.argv[3], "w") as outfile:

            identifier = mk_valid_identifier(ntpath.splitext(ntpath.basename(sys.argv[2]))[0]);
            outfile.write('#include <stddef.h>\n\n');
            outfile.write('static char {}[] =\n"'.format(identifier));

            while True:
                byte = infile.read(1)
                if byte == b"":
                    break
                write_byte(outfile, ord(byte))

            outfile.write('";\n\n');
            outfile.write('static const size_t {}_len = sizeof({}) - 1;\n'.format(identifier, identifier));

if __name__ == '__main__':
    main()
