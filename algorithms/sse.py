import heapq

class SpaceSaving:

    def __init__(self, capacity):
        self.capacity = capacity
        self.counters = {}
        self.min_heap = []

    def update(self, item):
        if item in self.counters:
            self.counters[item][0] += 1
            heapq.heappush(self.min_heap, (self.counters[item][0], item))
        elif len(self.counters) < self.capacity:
            self.counters[item] = [1, 0]
            heapq.heappush(self.min_heap, (self.counters[item][0], item))
        else:
            while self.min_heap:
                min_count, min_item = heapq.heappop(self.min_heap)
                if (min_item in self.counters and
                        self.counters[min_item][0] == min_count):
                    del self.counters[min_item]
                    self.counters[item] = [min_count + 1, min_count]
                    heapq.heappush(self.min_heap, (min_count + 1, item))
                    break

    def query(self, k=1):
        sorted_items = sorted(self.counters.items(),
                              key=lambda x: x[1][0],
                              reverse=True)
        return [(count_and_err[0], elem) for elem, count_and_err in sorted_items[:k]]

    def get_summary(self):
        print(f"当前容量: {self.capacity}, 已用: {len(self.counters)}")
        print("元素 -> [估计频数, 误差]")
        for elem, (cnt, err) in sorted(self.counters.items(),
                                        key=lambda x: x[1][0],
                                        reverse=True):
            print(f"  {elem:>10} -> [{cnt:>4}, {err:>4}]")
        print("-" * 30)
    