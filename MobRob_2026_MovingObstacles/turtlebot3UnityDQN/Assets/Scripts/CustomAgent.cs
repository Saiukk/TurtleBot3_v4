using UnityEngine;
using UnityEngine.AI;
using Unity.MLAgents;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using System.Linq;


// Declaration of the main class for the agent, that inherited from the Agent class of Unity-ML Agents
public class CustomAgent : Agent {

	// Robot physics parameters (angular and linear velocity constant)
	public float angularStep;
	public float linearStep;

	// Switches the action space from Discrete(3) (stop-and-rotate, used by DDQN/PPO)
	// to Continuous(2) (linear + angular velocity, blendable into arcs -- needed by
	// SAC/PPO for continuous control). Must be set before OnEnable() runs the
	// Agent's LazyInitialize(), which builds the actuator from BrainParameters -
	// hence overriding it here in Awake() rather than in the Agent's Initialize().
	public bool useContinuousActions = false;

	// Name of this agent's own target GameObject. Each agent looks up its target by
	// this name instead of a global tag, so two agents can chase two independent
	// targets (e.g. "Target1" / "Target2") in the same scene. Defaults to "Target"
	// to stay backwards compatible with the single-agent setup.
	public string targetName = "Target";

	// Must override (not shadow) the base Agent.Awake(): that base implementation is
	// what registers the ML-Agents communicator so the Editor can talk to Python at
	// all -- a plain (non-override) Awake() here would hide it entirely and silently
	// break every algorithm, not just SAC.
	protected override void Awake() {
		base.Awake();
		if (useContinuousActions) {
			GetComponent<BehaviorParameters>().BrainParameters.ActionSpec = ActionSpec.MakeContinuous(2);
		}
	}

	// Variables for the initial position of the target
	public bool randomizeAgentRotation = true;
	public bool randomizeAgentPosition = true;
	public bool randomizeTarget = true;
	public float targetRandomArea = 1.8f;
	public float distanceNormFact = 3.0f;

	// The object that represent the target
	private Transform target;

	// True once this agent has reached ITS OWN target. The episode is not over
	// until every agent in the scene has reached its own target, so a finished
	// agent freezes in place (kinematic, ignores further actions) instead of
	// keeping moving or being shoved around. Reset at every OnEpisodeBegin.
	private bool goalReached = false;

	// Cached Rigidbody, used to freeze the agent on its target (kinematic) so the
	// still-moving agent can't push it around while the episode keeps running.
	private Rigidbody rb;

	// The other agents in the scene (excluded: this agent itself), used for the
	// inter-agent observation, for collision handling and for spawn checks.
	private GameObject[] otherAgents;

	// The targets that do NOT belong to this agent (excluded: its own target), used
	// to keep the randomized spawn positions of the two targets apart.
	private Transform[] otherTargets;

	// Basic starting position/rotation of the agent for the reset
	// after every episode
	private Vector3 startingPos;
	private Quaternion startingRot;

	// Basic starting position/rotation of the target for the reset
	// after every episode
	private Vector3 startingPosTarget;
	private Quaternion startingRotTarget;

	// List of all the Obstacle to avoid
	private GameObject[] obstacleList;

	// List of all the Cost Area
	private GameObject[] costAreaList;

	// List of all the moving obstacles
	private MovingObstacle[] movingObstacles;

	// Reward support varaibles
	private float oldDistance;

	// Reusable buffer for the NavMesh path queries used to compute the geodesic distance
	private NavMeshPath navPath;


	// Called at the creation of the enviornment (before the first episode)
	// and only once
	public override void Initialize() {
		// Fill the target searching for the name given in the inspector first (so each
		// agent can own a distinct target), falling back to the global tag for the
		// legacy single-agent setup.
		GameObject targetObject = GameObject.Find(targetName);
		if (targetObject == null) targetObject = GameObject.FindGameObjectWithTag("Target");
		target = targetObject.transform;
		// Fill the list of the Obstacle searching for the tag (setted in the editor)
		obstacleList = GameObject.FindGameObjectsWithTag("Obstacle");
		// Fill the list of the Obstacle searching for the tag (setted in the editor)
		costAreaList = GameObject.FindGameObjectsWithTag("CostArea");
		// Fill the list of the moving obstacles searching for the class (setted in the editor)
		movingObstacles = GameObject.FindObjectsOfType<MovingObstacle>();
		// Every other agent in the scene (used for the inter-agent observation,
		// agent-agent collisions and non-overlapping spawn points)
		otherAgents = System.Array.FindAll(GameObject.FindGameObjectsWithTag("Agent"), o => o != gameObject);
		// Every target that is not this agent's own (keeps the two targets apart)
		otherTargets = System.Array.FindAll(
			GameObject.FindGameObjectsWithTag("Target"),
			o => o.transform != target
		).Select(o => o.transform).ToArray();
		// Cache the Rigidbody used to freeze the agent once it reaches its target
		rb = GetComponent<Rigidbody>();
		// Setting the basic rotation and position
		startingPos = transform.position;
		startingRot = transform.rotation;
		// Setting the basic rotation and position of the target
		startingPosTarget = target.transform.position;
		startingRotTarget = target.transform.rotation;
		// Compute the initial distance from the target
		oldDistance = Vector3.Distance( target.position, transform.position );
		navPath = new NavMeshPath();
	}


	// Called at the every new episode of the training loop,
	// after each reset (both from target, crash or timeout)
	public override void OnEpisodeBegin() {

		// Un-freeze the agent in case the previous episode ended with it parked on
		// its target (goalReached freezes the Rigidbody to make it immovable).
		goalReached = false;
		if (rb != null) rb.isKinematic = false;

		// Reset the position of the target to the basic settings
		foreach (MovingObstacle mo in movingObstacles) mo.ResetObstacle();
		// Randomize the position of the target, iterate check to avoid compenetration
		// between the target, the Obstacle, the other agents and the other targets
		//target.GetComponentInChildren<MeshRenderer>().enabled = false;
		if( randomizeTarget ) {
			do {
				target.position = new Vector3(Random.Range(-targetRandomArea, targetRandomArea), 0.0f, Random.Range(-targetRandomArea, targetRandomArea));	
			} while ( verifyIntersectionWithObstacle( target.gameObject ) || verifyIntersectionWithOtherAgents( target.gameObject ) || verifyIntersectionWithOtherTargets( target.gameObject ) );
		}
		// Reset the position of the agent to the basic settings
		// at the beginning of each episode
		transform.position = startingPos;
		transform.rotation = startingRot;
		// If the flag is active randomize the initial agent rotation at each episode
		if( randomizeAgentRotation ) transform.Rotate( new Vector3(0f, Random.Range(0, 360), 0f) );
		// If the flag is active randomize the initial agent position at each episode
		if( randomizeAgentPosition ) {
			do {
				transform.position = new Vector3(Random.Range(-targetRandomArea, targetRandomArea), 0.0f, Random.Range(-targetRandomArea, targetRandomArea));	
			} while ( verifyIntersectionWithObstacle( this.gameObject ) || verifyIntersectionWithOtherAgents( this.gameObject ) );
		}
		// Compute the initial distance from the target
		oldDistance = Vector3.Distance( target.position, transform.position );
	}


	// Listener for the action received, both from the neural network and the keyboard
	// (if heuristic mode), inside the Python script, the action is passed with the step funciton
	public override void OnActionReceived(ActionBuffers actionBuffers)	{

		// A finished agent idles at its own target: accept no more actions (Python
		// still sends them every step, Unity just ignores them) and don't move.
		// The episode only ends when EVERY agent reaches its own target.
		if (goalReached) return;

		float angularVelocity;
		float linearVelocity;

		if ( useContinuousActions ) {
			// gym_unity always exposes a continuous action space as Box(-1, 1), regardless
			// of the physical meaning assigned here, so both components arrive in [-1, 1].
			// Component 0 is clamped to [0, 1] (forward only, no reverse): the LiDAR only
			// covers a 180-degree forward-facing arc, so driving backward would be blind
			// to anything behind the robot. Turning (component 1) still uses the full
			// range so the agent can blend a turn with forward motion into an arc --
			// the discrete action set (below) can only ever do one or the other.
			var continuousAction = actionBuffers.ContinuousActions;
			linearVelocity = Mathf.Clamp01(continuousAction[0]) * linearStep;
			angularVelocity = Mathf.Clamp(continuousAction[1], -1f, 1f) * angularStep;
		} else {
			// Read the action buffer, in this set-up, discrete
			var actionBuffer = actionBuffers.DiscreteActions;
			// Basic setting for the action 0 (CoC)
			angularVelocity = 0f;
			linearVelocity = linearStep;
			// Listener for action 1, turn right
			// change angular and lienar velocity
			if ( actionBuffer[0] == 1 ) {
				angularVelocity = angularStep;
				linearVelocity = 0f;
			}
			// Listener for action 2, turn left
			// change angular and lienar velocity
			if ( actionBuffer[0] == 2 ) {
				angularVelocity = -angularStep;
				linearVelocity = 0f;
			}
		}

		// Apply the movement (rotation and translation) according with angular and linear velocity
		//transform.Rotate(Vector3.up * Time.deltaTime * angularVelocity);
		//transform.Translate(Vector3.forward * Time.deltaTime * linearVelocity);
		transform.Rotate(Vector3.up * angularVelocity);
		transform.Translate(Vector3.forward * linearVelocity);

		// Steps for agent and moving obstacles
		foreach (MovingObstacle mo in movingObstacles) mo.Step();
	}


	// Listener for the observations collections.
	// The observations for the LiDAR sensor are inherited from the 
	// editor, in thi function we add the other observations (angle, distance)
	public override void CollectObservations(VectorSensor sensor) {	

		// Compute the geodesic (obstacle-aware) distance between agent and target via the
		// baked NavMesh, instead of the straight-line distance: a straight-line metric
		// falsely penalizes the detour needed to enter a corridor whose opening isn't on
		// the direct line to the target, since that detour temporarily increases it.
		float distance = ComputeGeodesicDistance();
		// Normalization of the distance on the size of the room in [0, 1]. Geodesic paths
		// are longer than the straight-line distance, so clamp defensively.
		distance = Mathf.Clamp01( distance / distanceNormFact );
		// Compute the angle using the built-in function, the function returns a value between -180 and +180
		Vector3 targetDir = target.position - transform.position;
		float angle = Vector3.SignedAngle(targetDir, transform.forward, transform.up);
		// Normalize between [0, 1] also the angle
		angle = 0.5f - (angle / 360f);
		// Add the two observations inside the array of the obseravtions
		sensor.AddObservation( angle );
		sensor.AddObservation( distance );
		// Inter-agent observation: normalized angle and distance to the nearest other
		// agent, encoded exactly like the target observation above (same normalization
		// factors). This is what turns the task into a proper multi-agent system -- each
		// agent can perceive the other and learn to keep clear of it.
		GameObject otherAgent = NearestOtherAgent();
		if ( otherAgent != null ) {
			Vector3 otherDir = otherAgent.transform.position - transform.position;
			float otherAngle = Vector3.SignedAngle(otherDir, transform.forward, transform.up);
			otherAngle = 0.5f - (otherAngle / 360f);
			float otherDistance = Mathf.Clamp01( Vector3.Distance( otherAgent.transform.position, transform.position ) / distanceNormFact );
			sensor.AddObservation( otherAngle );
			sensor.AddObservation( otherDistance );
		} else {
			// No other agent in the scene: report the null-case (angle 0, max distance)
			sensor.AddObservation( 0f );
			sensor.AddObservation( 1f );
		}
		// Add the special observation for the cost, does not affect the training
		int costState = verifyIntersectionWithCostArea() ? 1 : 0;
		sensor.AddObservation( costState );
	}


	// Path length from the agent to the target along the baked NavMesh (requires a
	// NavMeshSurface baked over the static arena geometry). Falls back to the
	// straight-line distance if the query fails (e.g. momentarily off-mesh).
	private float ComputeGeodesicDistance() {
		if ( NavMesh.CalculatePath( transform.position, target.position, NavMesh.AllAreas, navPath )
			&& navPath.status == NavMeshPathStatus.PathComplete ) {

			float total = 0f;
			for ( int i = 0; i < navPath.corners.Length - 1; i++ )
				total += Vector3.Distance( navPath.corners[i], navPath.corners[i + 1] );
			return total;
		}

		return Vector3.Distance( transform.position, target.position );
	}


	// Debug function, useful to control the agent with the keyboard in heurisitc mode
	// (must be setted in the editor)
	public override void Heuristic(in ActionBuffers actionsOut) {

		if ( useContinuousActions ) {
			// W drives forward, A/D turn -- no reverse (see OnActionReceived: the LiDAR
			// can't see behind the robot). Turning zeroes the forward speed to mirror
			// the discrete heuristic below (turn-in-place rather than an arc), since
			// this is just a manual debug aid, not something training relies on.
			bool turning = Input.GetKey(KeyCode.A) || Input.GetKey(KeyCode.D);
			float forward = Input.GetKey(KeyCode.W) ? 1f : 0f;
			float turn = 0f;
			if (Input.GetKey(KeyCode.A)) turn = 1f;
			if (Input.GetKey(KeyCode.D)) turn = -1f;

			var continuousActionsOut = actionsOut.ContinuousActions;
			continuousActionsOut[0] = turning ? 0f : forward;
			continuousActionsOut[1] = turn;
		} else {
			// Set the basic action and wait or a keyboard key
			int action = 0;
			if (Input.GetKey(KeyCode.A)) action = 1;
			if (Input.GetKey(KeyCode.D)) action = 2;
			// Add the action to the actionsOut object
			var actions = actionsOut.DiscreteActions;
			actions[0] = action;
		}
	}

	
	// Listener for the collison with a solid object
	private void OnCollisionEnter(Collision collision) { 

		// Check if the collision is within an obstacle (avoid activation with the floor)
		// or with a wall, the end of the episode is now menaged by the wrapper.
		// Colliding with the other agent is treated exactly like hitting an obstacle
		// (crash): the two robots must learn to avoid each other.
		// The agent's colliders live on CHILD objects (CubeModel / ModelMaskResize)
		// tagged "Untagged", so the "Agent" tag can't be found on the collider itself:
		// climb to the root GameObject, which carries the "Agent" tag. The root of
		// the other agent is never this agent's own root, so no self-collision risk.
		// Set the reward base value for a crash
		if (collision.collider.CompareTag("Obstacle") || collision.collider.CompareTag("Wall")
			|| collision.collider.transform.root.CompareTag("Agent")) SetReward(-1f);
	}


	// Listener for the collison with a trigger (non-solid) object, the end of the 
	// episode is now menaged by the wrapper.
	private void OnTriggerStay(Collider collision) { 

		// Only THIS agent's own target counts as a success. Both targets share the
		// "Target" tag, so the tag alone can't tell them apart: touching the other
		// agent's target must NOT be rewarded. A small penalty keeps an agent from
		// lingering on the wrong target (0.4, not a multiple of the collision -1,
		// so it can never be misread as a crash).
		if (collision.CompareTag("Target") && collision.transform == target) {
			SetReward(1f);
			// Freeze the agent in place: it reached its goal but the episode keeps
			// running until the other agent does the same. Freezing (kinematic) both
			// stops its own motion and prevents the other agent from shoving it.
			if (!goalReached) {
				goalReached = true;
				if (rb != null) rb.isKinematic = true;
			}
		} else if (collision.CompareTag("Target")) {
			// The other agent's target: not a success for this agent.
			SetReward(-0.4f);
		}
	}


	// Utility function to check if there is an intersection between the input object
	// and one of the obstacles
	private bool verifyIntersectionWithObstacle( GameObject gO ) {
		// Iterate over the list of the Obstacle
		foreach( GameObject obstacle in obstacleList )
			if( obstacle.GetComponent<Renderer>().bounds.Intersects( gO.GetComponentInChildren<Renderer>().bounds ) ) 
				return true;
		return false;
	}


	// Utility function to check if there is an intersection between the input object
	// and one of the other agents (keeps agents from spawning on top of each other)
	private bool verifyIntersectionWithOtherAgents( GameObject gO ) {
		foreach( GameObject other in otherAgents )
			if( other.GetComponentInChildren<Renderer>().bounds.Intersects( gO.GetComponentInChildren<Renderer>().bounds ) ) 
				return true;
		return false;
	}


	// Utility function to check if there is an intersection between the input object
	// and one of the targets that do not belong to this agent (keeps the two targets
	// from spawning on top of each other)
	private bool verifyIntersectionWithOtherTargets( GameObject gO ) {
		foreach( Transform other in otherTargets )
			if( other.GetComponentInChildren<Renderer>().bounds.Intersects( gO.GetComponentInChildren<Renderer>().bounds ) ) 
				return true;
		return false;
	}


	// Nearest other agent in the scene (by squared distance), or null if none exists
	private GameObject NearestOtherAgent() {
		GameObject nearest = null;
		float bestSqr = float.MaxValue;
		foreach( GameObject other in otherAgents ) {
			float sqr = ( other.transform.position - transform.position ).sqrMagnitude;
			if( sqr < bestSqr ) { bestSqr = sqr; nearest = other; }
		}
		return nearest;
	}


	// Utility function to check if there is an intersection between the agent
	// and one of the Cost Area
	private bool verifyIntersectionWithCostArea( ) {

		// Iterate over the list of the Obstacle
		foreach( GameObject costArea in costAreaList ) {
			if( costArea.GetComponent<Renderer>().bounds.Intersects( GetComponentInChildren<Renderer>().bounds ) ) {
				costArea.GetComponent<MeshRenderer>().enabled = true;
				return true;
			}
			costArea.GetComponent<MeshRenderer>().enabled = true;
		}
		return false;
	}


}