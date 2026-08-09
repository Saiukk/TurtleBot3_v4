using UnityEngine;
using UnityEngine.AI;
using Unity.MLAgents;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;


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
		// Fill the game object target searching for the tag (setted in the editor)
		target = GameObject.FindGameObjectWithTag("Target").transform;
		// Fill the list of the Obstacle searching for the tag (setted in the editor)
		obstacleList = GameObject.FindGameObjectsWithTag("Obstacle");
		// Fill the list of the Obstacle searching for the tag (setted in the editor)
		costAreaList = GameObject.FindGameObjectsWithTag("CostArea");
		// Fill the list of the moving obstacles searching for the class (setted in the editor)
		movingObstacles = GameObject.FindObjectsOfType<MovingObstacle>();
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

		// Reset the position of the target to the basic settings
		foreach (MovingObstacle mo in movingObstacles) mo.ResetObstacle();
		// Randomize the position of the target, iterate check to avoid compenetration
		// between the target and the Obstacle
		//target.GetComponentInChildren<MeshRenderer>().enabled = false;
		if( randomizeTarget ) {
			do {
				target.position = new Vector3(Random.Range(-targetRandomArea, targetRandomArea), 0.0f, Random.Range(-targetRandomArea, targetRandomArea));	
			} while ( verifyIntersectionWithObstacle( target.gameObject ) );
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
			} while ( verifyIntersectionWithObstacle( this.gameObject ) );
		}
		// Compute the initial distance from the target
		oldDistance = Vector3.Distance( target.position, transform.position );
	}


	// Listener for the action received, both from the neural network and the keyboard
	// (if heuristic mode), inside the Python script, the action is passed with the step funciton
	public override void OnActionReceived(ActionBuffers actionBuffers)	{

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
		// Set the reward base value for a crash
		if (collision.collider.CompareTag("Obstacle") || collision.collider.CompareTag("Wall")) SetReward(-1f);
	}


	// Listener for the collison with a trigger (non-solid) object, the end of the 
	// episode is now menaged by the wrapper.
	private void OnTriggerStay(Collider collision) { 

		// Check collision with the target and set reward base value for a success
		if (collision.CompareTag("Target")) SetReward(1f);
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