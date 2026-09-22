/* Copyright 2019 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_HLO_ANALYSIS_HLO_REPLICATION_ANALYSIS_H_
#define XLA_HLO_ANALYSIS_HLO_REPLICATION_ANALYSIS_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/node_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/shape_tree.h"
#include "xla/xla_data.pb.h"

namespace xla {

// A wrapper around absl::Span<const ReplicaGroup> that allows us to hash it
class HashableReplicaGroupSpan : public absl::Span<const ReplicaGroup> {
 public:
  explicit HashableReplicaGroupSpan(const absl::Span<const ReplicaGroup> groups)
      : absl::Span<const ReplicaGroup>(groups) {}

  bool operator==(const HashableReplicaGroupSpan& other) const {
    if (size() != other.size()) {
      return false;
    }
    for (int i = 0; i < size(); ++i) {
      if (this->at(i).replica_ids().size() !=
          other.at(i).replica_ids().size()) {
        return false;
      }
      for (int j = 0; j < this->at(i).replica_ids().size(); ++j) {
        if (this->at(i).replica_ids()[j] != other.at(i).replica_ids()[j]) {
          return false;
        }
      }
    }
    return true;
  }

  template <typename H>
  friend H AbslHashValue(H h, const HashableReplicaGroupSpan& a) {
    for (const auto& group : a) {
      for (int64_t id : group.replica_ids()) {
        h = H::combine(std::move(h), id);
      }
    }
    return H::combine(std::move(h), a.size());
  }
};

// An HLO pass that determines whether each instruction in the module outputs
// the same value across replicas or across partitions (depending on the value
// `cross_partition_spmd`). It propagates sources of replicated values to
// the rest of the module, where sources include cross-replica-sum, annotated
// entry parameters, and constants.
class HloReplicationAnalysis {
 public:
  // Runs the analysis on module and returns the result or an error.
  static absl::StatusOr<std::unique_ptr<HloReplicationAnalysis>> Run(
      const HloModule* module, bool cross_partition_spmd);

  // Same as above, but the caller can provide additional annotations: a set of
  // while loops that are known to have the same iteration counts across
  // replicas or partitions.
  static absl::StatusOr<std::unique_ptr<HloReplicationAnalysis>> Run(
      const HloModule* module, bool cross_partition_spmd,
      const absl::flat_hash_set<const HloInstruction*>*
          loops_known_with_same_iterations);

  // Same as above but supports finding partially replicated HLOs.
  static absl::StatusOr<std::unique_ptr<HloReplicationAnalysis>>
  RunWithPartialReplication(const HloModule* module, bool cross_partition_spmd);

  // Returns if the HLO instruction outputs the same value (i.e., replicated) at
  // the given index across all replicas or partitions.
  bool HloInstructionIsReplicatedAt(const HloInstruction* inst,
                                    const ShapeIndex& index) const;

  bool HloInstructionIsReplicatedAt(
      const HloInstruction* inst, const ShapeIndex& index,
      absl::Span<const ReplicaGroup> replica_groups) const;

 private:
  // A data structure that represents how an HLO is replicated among a set of
  // devices. Device ID could be either partition ID or replica ID.
  // We represent partial replication by grouping devices that have the same
  // value into the same set.
  class HloReplication {
   public:
    static HloReplication ReplicatedOnAllDevices();
    static HloReplication UniqueOnAllDevices();
    static HloReplication PartiallyReplicated(
        absl::Span<const std::vector<std::vector<int64_t>>>
            device_sets_per_replica);
    HloReplication();
    HloReplication(const HloReplication& other) = default;
    HloReplication(HloReplication&& other) = default;
    HloReplication& operator=(HloReplication&& other) = default;
    HloReplication Merge(const HloReplication& other) const;
    bool Equal(const HloReplication& other) const;
    bool operator==(const HloReplication& rhs) const;
    bool IsReplicatedOnAllDevices() const;
    bool IsUniqueOnAllDevices() const;
    bool IsPartiallyReplicated() const;
    bool IsReplicatedWithinSubgroup(absl::Span<const int64_t> device_ids) const;
    std::string ToString() const;

    template <typename H>
    friend H AbslHashValue(H h, const HloReplication& r) {
      h = H::combine(std::move(h), r.state_);
      if (r.device_set_root_per_replica_ != nullptr) {
        h = H::combine(std::move(h), *r.device_set_root_per_replica_);
      }
      return h;
    }

   private:
    enum class State {
      kReplicatedOnAllDevices = 0,
      kUniqueOnAllDevices = 1,
      kPartiallyReplicated = 2,
    };
    // Only partially replicated values carry device sets; the other two states
    // share a null pointer so that copying them is a plain state copy.
    explicit HloReplication(State state);
    explicit HloReplication(
        absl::Span<const std::vector<int64_t>> device_set_root_per_replica);
    State state_;
    // Helper class that subclasses T, and computes the hash once on
    // construction, and intercepts the hash function to use the precomputed
    // hash.
    template <typename T>
    class HashOnConstruction : public T {
     public:
      template <typename V>
      explicit HashOnConstruction(V& device_set_root_per_replica)
          : T(device_set_root_per_replica.begin(),
              device_set_root_per_replica.end()),
            hash_(absl::HashOf(device_set_root_per_replica)) {}

      const size_t hash_;

      template <typename H>
      friend H AbslHashValue(H h, const HashOnConstruction& r) {
        return H::combine(std::move(h), r.hash_);
      }
    };
    // Null if state_ is kReplicatedOnAllDevices or kUniqueOnAllDevices.
    //
    // If cross_partition_spmd is true, device_set_root_per_replica_[k]'s size
    // equals the number of partitions, and within replica k,
    // device_set_root_per_replica_[k] maps each partition ID to the smallest
    // partition ID in the set.
    //
    // If cross_partition_spmd is false, device_set_root_per_replica_[k]'s size
    // equals the number of replicas, and within partition k,
    // device_set_root_per_replica_[k] maps each replica to the smallest replica
    // ID in the set.
    std::shared_ptr<const HashOnConstruction<std::vector<std::vector<int64_t>>>>
        device_set_root_per_replica_;
  };

  std::vector<std::vector<std::vector<int64_t>>> GroupsForReplicas(
      absl::Span<const ReplicaGroup> groups);

  HloReplication DetermineHloInstructionIsReplicated(const HloInstruction* hlo,
                                                     const ShapeIndex& index);

  HloReplication MergeReplications(const HloReplication& replication_a,
                                   const HloReplication& replication_b) {
    // Merging with a value that is replicated or unique on all devices is a
    // copy of one side; only merges of two partially replicated values are
    // worth memoizing.
    if (!replication_a.IsPartiallyReplicated() ||
        !replication_b.IsPartiallyReplicated()) {
      return replication_a.Merge(replication_b);
    }
    std::pair<HloReplication, HloReplication> key = {replication_a,
                                                     replication_b};

    // Look replication pair up in map: if not found we pass the pair to an
    // overloaded constructor of HloReplication which constructs and returns
    // a merged HloReplication.
    auto [iter, inserted] = replication_merge_map_.try_emplace(key);
    if (inserted) {
      iter->second = replication_a.Merge(replication_b);
    }
    return iter->second;
  }

  HloReplicationAnalysis(const HloModule* module, bool cross_partition_spmd,
                         const absl::flat_hash_set<const HloInstruction*>*
                             loops_known_with_same_iterations,
                         bool support_partial_replication)
      : module_(module),
        cross_partition_spmd_(cross_partition_spmd),
        loops_known_with_same_iterations_(*loops_known_with_same_iterations),
        support_partial_replication_(support_partial_replication),
        num_partitions_(module_->config().num_partitions()),
        replica_count_(module_->config().replica_count()) {}

  // Computes hlo_replication_.
  absl::Status ComputeHloReplication();

  // A helper function to recursively compute hlo_replication on a computation.
  // Returns whether hlo_replication_ is changed.
  bool ComputeHloReplicationOnComputation(const HloComputation* computation,
                                          bool mark_everything_not_replicated);

  // Returns the post order of `computation`. The module does not change during
  // the analysis, so the order is computed once per computation.
  const std::vector<HloInstruction*>& InstructionPostOrder(
      const HloComputation* computation);

  // Returns the replication of `inst`, which must have been computed already.
  const ShapeTree<HloReplication>& GetReplication(
      const HloInstruction* inst) const;

  // Merges `source` into `dest` element by element. Returns whether anything
  // changed.
  bool CombineReplication(const ShapeTree<HloReplication>& source,
                          ShapeTree<HloReplication>* dest);

  // Assigns `replication` to `dest` if it has none yet, or combines it with
  // the existing one. Returns whether anything changed.
  bool AssignOrCombineReplication(ShapeTree<HloReplication> replication,
                                  const HloInstruction* dest);

  // Assigns or combines the replication of `source` to `dest`. Returns whether
  // anything changed; nothing changes if `source` has no replication yet.
  bool PropagateReplication(const HloInstruction* source,
                            const HloInstruction* dest);

  // Marks `inst` unique on all devices at every index. Returns whether
  // anything changed.
  bool MarkNotReplicated(const HloInstruction* inst);

  // Builds the replica group dedup map that allows caching replication
  // calculations for all-reduce/all-gather that share the same replica groups.
  // This can significantly help in compile times when replica groups are very
  // large.
  void BuildReplicaGroupDedupMap();

  const HloModule* module_;

  // If true, run this replication analysis for replicated values across
  // partitions (not across replicas) on an SPMD partitioned module. This means
  // that HloInstructionIsReplicatedAt() returns true if the value is identical
  // across partitions for each replica. The module-level parameter and root
  // instructions may have HloSharding attributes that indicate whether values
  // are identical across partitions.
  //
  // If false, HloReplicationAnalysis runs across replicas.
  const bool cross_partition_spmd_;

  // A set of while loops that are known to have the same iteration counts
  // across replicas or partitions. This is provided by the caller as additional
  // annotations.
  const absl::flat_hash_set<const HloInstruction*>&
      loops_known_with_same_iterations_;

  const bool support_partial_replication_;

  // Capture the number of partitions / replicas for the module.
  const int64_t num_partitions_, replica_count_;

  // A map from each analyzed HLO instruction to a shape tree that represents
  // whether the instruction outputs the same value across replicas or
  // partitions at each shape index. Trees built during the walk point at an
  // instruction's shape instead of copying it (a propagated tree points at the
  // shape of the instruction it was copied from). Never read such a tree's
  // shape() after the analysis has run: HloInstruction::mutable_shape() and
  // shape canonicalization replace the pointed Shape object, and callers
  // delete instructions while they hold the analysis. Queries use only the
  // index table and the node values, so they stay valid.
  absl::flat_hash_map<const HloInstruction*, ShapeTree<HloReplication>>
      hlo_replication_;

  // Post order of every computation analyzed so far. The values must not move
  // when the map grows, since a visit of a nested computation adds an entry
  // while the caller iterates over its own post order.
  absl::node_hash_map<const HloComputation*, std::vector<HloInstruction*>>
      post_orders_;

  // Computations already visited with mark_everything_not_replicated set.
  // That visit marks the parameters and every ordinary instruction unique;
  // tuples, get-tuple-elements, optimization barriers and nested calls derive
  // from those values and infeeds from their sharding, and replication only
  // moves towards unique, so a repeat visit with the flag set would recompute
  // the same values and is skipped.
  absl::flat_hash_set<const HloComputation*>
      computations_marked_not_replicated_;

  // Replications for all-reduce/all-gather that have the same replica groups is
  // usually identical. We use the following data structures to memoize the
  // replications for instructions with identical replica groups.
  absl::flat_hash_map<const HloInstruction*, std::optional<HloReplication>*>
      replica_group_dedup_map_;
  absl::flat_hash_map<std::pair<HloReplication, HloReplication>, HloReplication>
      replication_merge_map_;
  std::vector<std::optional<HloReplication>> unique_replications_;
  absl::flat_hash_map<HashableReplicaGroupSpan,
                      std::vector<std::vector<std::vector<int64_t>>>>
      device_sets_per_replica_map_;
};

}  // namespace xla

#endif  // XLA_HLO_ANALYSIS_HLO_REPLICATION_ANALYSIS_H_
