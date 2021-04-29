class Solution(object):
    def nextPermutation(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        n = len(nums)
        if n == 0 or n == 1:
            return nums

        right_read = [nums[-1]]  # read in elts of nums from right to left
        for i, num in enumerate(reversed(nums[:-1])):
            j = n - i - 2
            if num < right_read[-1]:  # it's descending
                nums[j:] = splice(right_read, num)
                return None
            else:
                right_read.append(num)

        nums.reverse()

def splice(ascending_nums, x):
    output = []
    next_biggest = -1
    for i, num in enumerate(ascending_nums):
        if num > x and next_biggest == -1:
            next_biggest = num
            output.append(x)
        else:
            output.append(num)

    return [next_biggest] + output

if __name__ == '__main__':
    nums = [1,2,3,4,5,8,7,6]
    # nums = [2,3,1]
    foo = Solution()
    foo.nextPermutation(nums)
    print(nums)