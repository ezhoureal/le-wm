## Long Horizon
- For intermediate rollouts, use subgoals instead of final goal to compute cost
- Generate subgoals through neural net, or hierarchical planning

## More Efficient Planning
- Action Instinct: Train an actor to predict the initial distribution for CEM to sample from

## LLM-guided subgoal planning to solve long horizon

**LLM produces structured state / subgoal specs → simulator renders them → JEPA encodes rendered image.**

For PushT, that means something like:

```text
subgoal:
  pusher_xy = (x, y)
  block_pose = (x, y, theta)
  target_pose = ...
  contact_side = "upper-right edge"
  phase = "approach / contact / push / align"
```

Then Gym / simulator renders this into the same observation distribution that le-WM was trained on. If the LLM outputs **proprioceptive / symbolic / geometric subgoals**, and the simulator renders them, you get:

* observations on the same visual distribution as training;
* physically valid object geometry;
* exact controllable coordinates;
* easy validation and correction;
* easier conversion into costs for planning.

So the LLM should act more like a **high-level task decomposer / waypoint proposer**, not as the final visual renderer.

### Architecture

I’d structure it like this:

```text
Task / goal
   ↓
LLM or VLM prior
   ↓
Structured subgoals: object pose, pusher pose, contact region, phase
   ↓
Gym renderer
   ↓
Rendered waypoint images
   ↓
JEPA encoder
   ↓
Latent-space planner / le-WM rollout / MPC
```


For example, for PushT:

```text
1. Move pusher near upper-right side of T.
2. Contact the green/T object from right side.
3. Push diagonally down-left.
4. Rotate object toward target orientation.
5. Fine-align center and angle.
```

Each step becomes a rendered state or latent goal.


### Algorithm
```
while (step < max_steps):
    next_subgoal_structure = LLM(current state, previous state, goal state)
    next_subgoal_image = gym_renderer(next_subgoal_structure)
    subgoal_latent = jepa.encode(next_subgoal_image)

    for step in receding_horizon:
        CEM planning -> JEPA predict + cost calculation
    execute actions
```

### LLM output format
```
{
  "phase": "approach_contact",
  "object_relation": "pusher is right of block",
  "desired_contact_side": "upper_right_edge",
  "next_subgoal": {
    "pusher_xy": [215, 130],
    "block_pose_delta": {
      "dx": -10,
      "dy": 5,
      "dtheta": -0.15
    }
  },
  "rationale": "approach the upper-right edge to create a diagonal push that rotates and translates the T toward the target"
}
```