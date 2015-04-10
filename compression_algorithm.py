"""
Just some playing around with utilities
"""

# enforces limits on indexes
def limited_index(max, index, width):
    start = 0 if width > index else index - width
    end = max if index + width > max else index + width
    return start, end

# gives a slice of an array in form of a generator
def slice_generator(array, index, width):
    start, end = limited_index(max=len(array), index=index, width=width)
    for item in array[start:end]:
        yield item


if __name__ == "__main__":
    l = [x for x in range(20)]
    print l
    for x in range(20):
        sg = slice_generator(l, x, 5)
        for y in sg:
            print y
        print "BREAK"

