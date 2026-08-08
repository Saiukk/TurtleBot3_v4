using UnityEngine;

public class MovingObstacle : MonoBehaviour {

	[Header("Waypoints del percorso (solo X/Z: la Y e' fissata da 'height')")]
	public Vector3[] waypoints;
	public bool loopClosed = true;   // dall'ultimo waypoint torna al primo

	// Static walls/obstacles in the scene sit at Y=0.1 (see their Transform + the
	// RayPerceptionSensor's StartVerticalOffset of 0.1, which casts rays at that
	// height): waypoints previously carried Y=0, so this object's collider sat
	// 0.1 units lower than everything else, right at the edge of the ray height
	// instead of centered in it -- the LiDAR essentially never registered a hit.
	// Forcing height here means waypoint authoring only has to get X/Z right.
	[Header("Altezza (deve combaciare con muri/ostacoli statici, Y=0.1)")]
	public float height = 0.1f;

	[Header("Velocita' in unita' per step dell'agente")]
	public float minSpeed = 0.005f;
	public float maxSpeed = 0.015f;

	private int currentIndex;   // segmento corrente: da waypoints[i] a waypoints[i+1]
	private float t;            // progresso sul segmento [0,1]
	private float speed;        // unita' mondo per step

	// Chiamata dall'agente a ogni reset di episodio
	public void ResetObstacle() {
		if (waypoints == null || waypoints.Length < 2) return;
		currentIndex = Random.Range(0, SegmentCount());
		t = Random.value;
		speed = Random.Range(minSpeed, maxSpeed);
		Apply();
	}

	// Chiamata dall'agente a ogni decision step
	public void Step() {
		if (waypoints == null || waypoints.Length < 2) return;

		float segLen = Vector3.Distance(waypoints[currentIndex], NextPoint());
		t += speed / Mathf.Max(segLen, 0.0001f);

		while (t >= 1f) {
			t -= 1f;
			currentIndex++;
			if (currentIndex >= SegmentCount()) currentIndex = 0;
			segLen = Vector3.Distance(waypoints[currentIndex], NextPoint());
		}
		Apply();
	}

	private int SegmentCount() {
		return loopClosed ? waypoints.Length : waypoints.Length - 1;
	}

	private Vector3 NextPoint() {
		return waypoints[(currentIndex + 1) % waypoints.Length];
	}

	private void Apply() {
		Vector3 pos = Vector3.Lerp(waypoints[currentIndex], NextPoint(), t);
		pos.y = height;
		transform.position = pos;
	}

	// Disegna il percorso nella Scene view (utile per verificare che non attraversi muri)
	void OnDrawGizmos() {
		if (waypoints == null || waypoints.Length < 2) return;
		Gizmos.color = Color.cyan;
		int n = loopClosed ? waypoints.Length : waypoints.Length - 1;
		for (int i = 0; i < n; i++) {
			Gizmos.DrawLine(waypoints[i], waypoints[(i + 1) % waypoints.Length]);
			Gizmos.DrawWireSphere(waypoints[i], 0.05f);
		}
	}
}