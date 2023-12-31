#ifndef VRAF_H
#define VRAF_H

#include <vector>
#include <string>
#include <fstream>
#include "nvmath/nvmath.h"
#include "../imgui/imgui.h"
#include "json.hpp"

using vec2 = nvmath::vec2f;
using vec3 = nvmath::vec3f;
using vec4 = nvmath::vec4f;

// Vector Recording and Filtering namespace
namespace VRaF {
	class Sequencer;

	struct Event {
		mutable int time;  // Start time, to be precise
		mutable int duration;
		// pair<float, float> is Time, Value
		// In keyframes, time is a float from 0 to 1; in order to ease scaling
		std::vector<std::pair<float, float>> keyframes;
		bool update(int frame);
		void filter(bool is_backwards);
		void filter();
		void clear();
		float* target = 0;
	};

	struct Recording {
		float* target;
		// As contrary to Event keyframes, this array holds
		// frame index as the key, thus the key is of type int
		// pair<int, float> is Frame, Value
		std::vector<std::pair<int, float>> keyframes;
		void update(int frame);
	};

	struct Track {
		ImColor color = ImColor::HSV(0.5f, 1.0f, 1.0f);
		std::vector<Event> events;
		std::vector<Recording> recordings;
		std::string label;
        std::function<void()> callback;
		bool is_expanded = true;
	};

	struct SeqState {
		bool isPlaying;
		float startTime;
		float currTime;
		int frame;
		ImVec2 zoom;
		ImVec2 pan;
		int range[2];
	};

	struct Dimentions {
		ImVec2 X;
		ImVec2 A;
		ImVec2 B;
		ImVec2 C;
		ImVec2 windowSize;
		float titlebarHeight;
	};

	enum SectionType {
		SECTION_CROSS,
		SECTION_LISTER,
		SECTION_TIMELINE,
		SECTION_EDITOR,

		SECTION_COMMON
	};

    class SeqIterator
    {
	public:
        SeqIterator(int frame, Sequencer* target);
		bool operator==(const SeqIterator& other) const {return frame == other.frame && target == other.target;}
		bool operator!=(const SeqIterator& other) const {return frame != other.frame || target != other.target;}
		SeqIterator operator++();
		SeqIterator operator++(int);
		int operator*() {return frame;}
	private:
		int frame;
		Sequencer* target;
    };

	class Sequencer
	{
	public:
		friend class SeqIterator;

		Sequencer(int fps=30);
		void toggle();
		void draw();
		void update(float time);
		SeqIterator begin();
		SeqIterator end();

		// The target must be contained in an event first.
		// If the event contains multiple targets, all of them will be recorded
		void record(float* target);
		void clear();
		void addKeyframe(std::string label, float step, int nsteps, float value, int step_start=0);
		void track(std::string label, float* value);
		void track(std::string label, vec2* value);
		void track(std::string label, vec3* value);
		void track(std::string label, vec4* value);
		void track(std::string label, float* value, std::function<void()> callback);
		void track(std::string label, vec2* value, std::function<void()> callback);
		void track(std::string label, vec3* value, std::function<void()> callback);
		void track(std::string label, vec4* value, std::function<void()> callback);
        void onFrameUpdated(std::function<void(int)> callback);

		void loadFile(std::string path);
		void saveFile(std::string path);

	private:
		SeqState state;
		std::vector<Track> tracks;
		int fps;
        std::function<void(int)> callback;

		Dimentions dims;
		void stop_recording();
		void drawBackground(SectionType section);
		void drawTracks(SectionType section);
		void drawGrid(SectionType section);
		void drawIndicators();
		void updateEvents();
		void updateEvents(int frame);
		ImFont* icons;
		ImFont* labels;
	};
}

#endif
